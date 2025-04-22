import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torch.nn import Module, Linear, BatchNorm2d, ReLU
from torch.nn import Conv2d
from torch.nn import BatchNorm2d as get_norm
from torch.nn.modules.utils import _pair
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from models.MobileNetV2_X5 import mobilenet_v2


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, leaky_relu=False, use_bn=True, frozen=False, prelu=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                self.bn = BatchNorm2d(nOut)
        else:
            self.bn = None

        if use_relu:
            if leaky_relu is True:
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif prelu is True:
                self.act = nn.PReLU(nOut)
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ResidualConvBlock, self).__init__()
        self.conv = ConvBNReLU(nIn, nOut, ksize=ksize, stride=stride, pad=pad,
                               dilation=dilation, groups=groups, bias=bias,
                               use_relu=use_relu, use_bn=use_bn, frozen=frozen)
        self.residual_conv = ConvBNReLU(nIn, nOut, ksize=1, stride=stride, pad=0,
                               dilation=1, groups=groups, bias=bias,
                               use_relu=False, use_bn=use_bn, frozen=frozen)

    def forward(self, x):
        x = self.conv(x) + self.residual_conv(x)
        return x


class ReceptiveConv(nn.Module):
    def __init__(self, inplanes, planes, baseWidth=24, scale=4, dilation=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, self.width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width*scale)
        #self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        for i in range(self.nums):
            self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3,
                    padding=dilation[i], dilation=dilation[i], bias=False))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        #if self.scale > 1:
        #    out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out


class InvertedResidual(nn.Module): 
    def __init__(self, inp, oup, stride=1, expand_ratio=4, residual=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, ksize=1, pad=0))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):  # x [1,320,10,10]
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class AirSOD(nn.Module):
    def __init__(self, args=None, pretrained=True,
                 enc_channels=[16, 24, 32, 96, 320],   
                 dec_channels=[16, 24, 32, 96, 320]): 
        super(AirSOD, self).__init__()
        # self.ms = args.ms
        self.ms = False  # make self.ms = False if doing the 'speed_test'

        print("Using AirSOD_V5")
        "****** AMC:Hybrid backbone *********"
        self.backbone = mobilenet_v2(pretrained)
        self.depthnet = DepthNet()

        "***** RIF: Recurrent Interaction Fusion module ****"
        self.down_conv3 = ConvBNReLU(nIn=64, nOut=32, ksize=1, stride=1, pad=0, )
        self.IF3 = InteractionFusion(in_planes=enc_channels[2], out_planes=dec_channels[2], groups=8, kernel_size=40, first=True)

        self.up_conv4 = ConvBNReLU(nIn=dec_channels[2], nOut=dec_channels[3], ksize=1, stride=2, pad=0)
        self.IF4 = InteractionFusion(in_planes=enc_channels[3], out_planes=dec_channels[3], groups=8, kernel_size=20, width=True)

        self.up_conv5 = ConvBNReLU(nIn=dec_channels[3], nOut=dec_channels[4], ksize=1, stride=2, pad=0)
        self.IF5 = InteractionFusion(in_planes=enc_channels[4], out_planes=dec_channels[4], groups=8, kernel_size=10)

        "***** Decoder *****"
        self.MFR = MFRDecoder(enc_channels, dec_channels)

        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)  # input_channel, output_channel, kernel_size
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)
        self.cls5 = nn.Conv2d(dec_channels[4], 1, 1, stride=1, padding=0)

    def loss(self, input, target):
        pass

    def forward(self, input, depth=None, test=True):
        # generate backbone features
        conv1, conv2, conv3, conv4, conv5 = self.backbone(input)  # [b,16,160,160],[b,24,80,80],[b,32,40,40],[b,96,20,20],[b,320,10,10]
        depth_features = self.depthnet(depth)

        if self.ms ==True:
            conv3 = F.interpolate(conv3,size=(40,40), mode='bilinear',align_corners=False)
            conv4 = F.interpolate(conv4, size=(20, 20), mode='bilinear', align_corners=False)
            conv5 = F.interpolate(conv5, size=(10, 10), mode='bilinear', align_corners=False)
            depth_features[-3] = F.interpolate(depth_features[-3], size=(40,40), mode='bilinear',align_corners=False)
            depth_features[-2] = F.interpolate(depth_features[-2], size=(20, 20), mode='bilinear', align_corners=False)
            depth_features[-1] = F.interpolate(depth_features[-1], size=(10, 10), mode='bilinear', align_corners=False)

        # RGB-D fuse
        conv3 = self.IF3(conv3, self.down_conv3(depth_features[-3]))
        conv4 = self.IF4(conv4, depth_features[-2], last_fea = self.up_conv4(conv3))
        conv5 = self.IF5(conv5, depth_features[-1], last_fea = self.up_conv5(conv4)) 

        # decoder
        features = self.MFR([conv1, conv2, conv3, conv4, conv5]) 
        saliency_maps = []
        for idx, feature in enumerate(features[:5]):
            saliency_maps.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1))(feature),
                    input.shape[2:],
                    mode='bilinear',
                    align_corners=False) 
            ) 
            if test:
                break
        saliency_maps = torch.sigmoid(torch.cat(saliency_maps, dim=1))
        return saliency_maps


# Criss-Cross Attention
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # [2,64,5,6]\[1,64,40,40]
        m_batchsize, _, height, width = x.size()  # batch, height, width
        proj_query = self.query_conv(x)  # Q
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,1)  
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,1)

        proj_key = self.key_conv(x)  # [1,8,40,40] K
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)  # [40,8,40]
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)  # [40,8,40]

        proj_value = self.value_conv(x)  # [1,64,40,40] V
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)  # [40,64,40]
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)  # [40,64,40]
        # [12,5,8]*[12,8,5]=[12,5,5],->view->[2,5,6,5]

        # energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,height,height).permute(0,2,1,3)  # (1,40,40,40)
        energy_H = torch.bmm(proj_query_H, proj_key_H).view(m_batchsize, width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width,width)  # [10,6,8]*[10,8,6]=[10,6,6],->view->[2,5,6,6]  # (1,40,40,40)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,height)  # [40,40,40]
        # todo Aggregation
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width,width)  # [40,40,40]
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3,1)  # [1,64,40,40]
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1,3)  # [1,64,40,40]
        return self.gamma * (out_H + out_W)  #  + x

# shifted MLP
class fea_split(nn.Module):
    def __init__(self,in_chans=3, embed_dim=768, kernel_size=7, stride=4, padding =0):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim//2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x): 
        x = self.proj(x)  

        x_split = torch.chunk(x, 2, 1)
        x1 = x_split[0]
        x2 = x_split[1]
        x1 = self.norm(x1)
        x2 = self.norm(x2)

        return x, x1, x2

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):  # x:(b,1024,160)
        # B, N, C = x.shape  # b:8, N:1024, C:160
        # x = x.transpose(1, 2).view(B, C, H, W)  # x:(b,160,32,32)
        x = self.dwconv(x)  # x:(b,160,32,32)
        # x = x.flatten(2).transpose(1, 2)  # x:(b,1024,160)

        return x

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1,padding=0, bias=False)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer() 
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0, bias=False)
        # self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):  # x(b,1024,160),H:32,W:32 /x(b,256,256), H:16,W:16
        # pdb.set_trace()
        B, C, H, W = x.shape

        # todo Shifted MLP (Width)
        # xn = x.transpose(1, 2).view(B, C, H, W).contiguous()  
        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)  
        xs = torch.chunk(xn, self.shift_size, 1)  # xs[(b,32,36,36),(b,32,36,36),(b,32,36,36),(b,32,36,36),(b,32,36,36)]
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))] 
        x_cat = torch.cat(x_shift, 1)  # (b,160,36,36) 
        x_cat = torch.narrow(x_cat, 2, self.pad, H)  # x_cat:(b,160,32,36)  torch.narrow(input, dim, start, length)
        x_s = torch.narrow(x_cat, 3, self.pad, W)  # x_s:(b,160,32,32)

        # x_s = x_s.reshape(B, C, H * W).contiguous()
        # x_shift_r = x_s.transpose(1, 2)  # x_shift_r:[b,1024,160]

        # todo project,
        x = self.fc1(x_s) 
        x = self.dwconv(x) 
        x = self.act(x) 
        # x = self.drop(x)

        # todo Shifted MLP (Height)
        # xn = x.transpose(1, 2).view(B, C, H, W).contiguous()  # xn:(b,160,32,32)
        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)  # xn:(b,160,36,36)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))] 
        x_cat = torch.cat(x_shift, 1)  # x_cat:(b,160,36,36)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)  # x_cat:(b,160,32,36)
        x_s = torch.narrow(x_cat, 3, self.pad, W)  # x_s:(b,160,32,32)
        # x_s = x_s.reshape(B, C, H * W).contiguous()  # x_s:[b,160,1024]
        # x_shift_c = x_s.transpose(1, 2)  # x_shift_c:[b,1024,160]

        # todo Reproject
        x = self.fc2(x_s) 
        # x = self.drop(x)
        return x

class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)  # LN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = self.drop_path(self.mlp(x))
        return x

# Depth Steam
class DepthNet(nn.Module):
    def __init__(self, pretrained=None, use_gan=False):
        super(DepthNet, self).__init__()
        block = InvertedResidual
        input_channel = 1
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 2, 2, 1],
            [4, 32, 2, 2, 1],
            [4, 64, 2, 2, 1],
        ]  # [4, 96, 2, 2, 1], [4, 320, 2, 2, 1],

        "************Tok-MLP start***************"
        drop_path_rate = 0.
        depths = [1, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        embed_dims = [64, 96, 320]
        num_heads = [1, 2, 4, 8]
        qkv_bias = False
        drop_rate = 0.
        attn_drop_rate = 0.
        sr_ratios = [8, 4, 2, 1]
        qk_scale = None

        self.cca3 = CrissCrossAttention(in_dim=embed_dims[1] // 2)
        self.conv3 = ConvBNReLU(nIn=embed_dims[1], nOut=embed_dims[1], ksize=1, pad=0)
        self.fea_split3 = fea_split(in_chans=embed_dims[0], embed_dim=embed_dims[1],kernel_size=1, stride=2)
        self.block3 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1] // 2, num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],sr_ratio=sr_ratios[0])])
        self.norm3 = nn.BatchNorm2d(embed_dims[1] // 2)

        self.cca4 = CrissCrossAttention(in_dim=embed_dims[2] // 2)
        self.conv4 = ConvBNReLU(nIn=embed_dims[2], nOut=embed_dims[2], ksize=1, pad=0)
        self.fea_split4 = fea_split(in_chans=embed_dims[1], embed_dim=embed_dims[2],kernel_size=1, stride=2)
        self.block4 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2] // 2, num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],sr_ratio=sr_ratios[0])])
        self.norm4 = nn.BatchNorm2d(embed_dims[2] // 2)
        "************Tok-MLP end***************"

        features = []
        # building inverted residual blocks  
        for t, c, n, s, d in inverted_residual_setting:
            output_channel = int(c * 1.0)
            for i in range(n):
                stride = s if i == 0 else 1
                dilation = d if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*features)

    def forward(self, x):
        feats = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in [1, 3, 5]:
                feats.append(x)

        "**************Attention-MLP Combined block Start***************"
        out = feats[-1]  # out [b,64,40,40]
        x, x1, x2 = self.fea_split3(out)  # out[b,400,128], H:20, W:20
        x2 = self.cca3(x2)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1)  # out[b,400,128]
        x1 = self.norm3(x1)  # out[b,400,128]
        out = self.conv3(torch.cat((x1 , x2), dim=1)) + x
        feats.append(out)

        x, x1, x2 = self.fea_split4(out)  # out[b,100,320], H:16, W:16
        x2 = self.cca4(x2)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1)  # out[b,100,320]
        x1 = self.norm4(x1)  # out[b,100,320]
        out = self.conv4(torch.cat((x1, x2), dim=1)) + x
        feats.append(out)
        "**************Attention-MLP Combined Elock end***************"
        return feats


class InteractionFusion(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False, first=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(InteractionFusion, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups  
        self.group_planes = out_planes // groups  # 
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        self.first = first

        "**********Bi-reference start*************"
        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())
        "**********Bi-reference end*************"

        # Multi-head self attention
        self.q_transform = nn.Conv1d(in_planes, out_planes // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_q = nn.BatchNorm1d(out_planes // 2)

        self.k_transform = nn.Conv1d(in_planes, out_planes // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_k = nn.BatchNorm1d(out_planes // 2)

        self.v_transform = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1,padding=0, bias=False)
        self.bn_v = nn.BatchNorm1d(out_planes)

        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))  #
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x, x_depth, last_fea=None):  # [1,320,10,10] [0B,1C,2H,3W]
        if self.first:
            x_f = x * x_depth
        else:
            x_f = last_fea

        "**********Bi-reference start*************"
        ca_rgb = self.channel_attention_rgb(self.squeeze_rgb(x))  # RGB channel attention
        ca_depth = self.channel_attention_depth(self.squeeze_depth(x_depth))  # Depth channel attention
        Co_ca3 = torch.softmax(ca_rgb + ca_depth, dim=1)

        x = x * (Co_ca3 + ca_rgb).expand_as(x)
        x_depth = x_depth * (Co_ca3 + ca_depth).expand_as(x_depth)
        "**********Bi-reference end*************"

        if self.width:  # x(N, C, H, W)
            x = x.permute(0, 2, 1, 3)  # N, H, C, W 
            x_depth = x_depth.permute(0, 2, 1, 3)
            x_f = x_f.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H [1,10,320,10]
            x_depth = x_depth.permute(0, 3, 1, 2)  # N, W, C, H [1,10,320,10] 
            x_f = x_f.permute(0, 3, 1, 2)  # N, W, C, H [1,10,320,10]
        N, W, C, H = x.shape  # N, W:10, C:320, H:10
        x = x.contiguous().view(N * W, C, H)  # [1*10,320,10]
        x_depth = x_depth.contiguous().view(N * W, C, H)  # [1*10,320,10]
        x_f = x_f.contiguous().view(N * W, C, H)  # [1*10,320,10]

        # Transformations
        q = self.bn_q(self.q_transform(x)).reshape(N * W, self.groups, self.group_planes // 2, H)  # (10,8,20,10)
        k = self.bn_k(self.k_transform(x_depth)).reshape(N * W, self.groups, self.group_planes // 2, H)   # (10,8,20,10)
        v = self.bn_v(self.v_transform(x_f)).reshape(N * W, self.groups, self.group_planes, H)   # (10,8,40,10)

        # Calculate position embedding 
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        # 上句q_embedding:[20,10,10],k_embedding:[20,10,10],v_embedding:[40,10,10]
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)  # q[10,8,20,10],q_embedding:[20,10,10] -> qr[10,8,10,10]
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)  # k[10,8,20,10]k_embedding:[20,10,10]->kr[10,8,10,10]
        qk = torch.einsum('bgci, bgcj->bgij', q, k)  # q[10,8,20,10],k[10,8,20,10]->qk[10,8,10,10]

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)  # stacked_similarity[10,24,10,10]
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)  # stacked_similarity[10,8,10,10]
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)  # similarity[10,8,10,10]
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)  # sv [10,8,40,10]
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)  # sve [10,8,40,10]
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)  # [10,640,10]
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)  # [1,10,320,10]

        if self.width:
            output = output.permute(0, 2, 1, 3)  # [1,320,10,10]
        else:
            output = output.permute(0, 2, 3, 1)  # [1,320,10,10]

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        # self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        self.q_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        self.k_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        self.v_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

# todo MFR
"*******MFR: Multi-branch Feature Refinement start*********"
class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size,hidden_dims = [16,24,32,96,320], stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=3, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob

        self.use_bn = norm is not None
        if self.use_bn:
            self.bn0 = get_norm(norm, channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = get_norm(norm, channels)
        self.fc2 = Conv2d(channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)  # radix:2, groups:1
        "***********multi-scale start***********"
        dilation = [1, 2, 3]
        self.hidden_conv1 = nn.ModuleList()
        self.hidden_conv2 = nn.ModuleList()
        self.hidden_conv3 = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.hidden_conv1.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i], kernel_size=3, padding=dilation[0], groups=hidden_dims[i],
                          dilation=dilation[0]),
                nn.BatchNorm2d(hidden_dims[i]),
                nn.ReLU(inplace=True)))

            self.hidden_conv2.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i], kernel_size=3, padding=dilation[1], groups=hidden_dims[i],
                          dilation=dilation[1]),
                nn.BatchNorm2d(hidden_dims[i]),
                nn.ReLU(inplace=True)))

            self.hidden_conv3.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i], kernel_size=3, padding=dilation[2], groups=hidden_dims[i],
                          dilation=dilation[2]),
                nn.BatchNorm2d(hidden_dims[i]),
                nn.ReLU(inplace=True)))
        "***********multi-scale end*************"

    def forward(self, x, idx=None):
        batch, rchannel = x.shape[:2]  # batch:1, rchannel:320
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)  # splited(tuple:2)([1,160,10,10],[1,160,10,10])
            "********multi-scale start**********"
            m1 = self.hidden_conv1[idx](splited[0])
            m2 = self.hidden_conv2[idx](splited[1])
            m3 = self.hidden_conv3[idx](splited[2])
            ms_splited = (m1,m2,m3)
            "********multi-scale end************"
            gap = sum(splited)  # gap[1,160,10,10]
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)  # gap[1,160,1,1]
        gap = self.fc1(gap)  # 
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)  # 
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1) 

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)  # attens(tuple:2)([1,160,10,10],[1,160,10,10])
            out = sum([att*split for (att, split) in zip(attens, ms_splited)])  # out[1,160,10,10]
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class MFB(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims, expansion=4, input_num=2):
        super(MFB, self).__init__()
        self.fuse = SplAtConv2d(in_channels, out_channels, kernel_size=1, hidden_dims=hidden_dims)

    def forward(self, cat_feat, idx=None):  # low(48,20,20),high(48,10,10)
        final = self.fuse(cat_feat, idx)
        return final

class MFRDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFRDecoder, self).__init__()
        print("Using MFRDecoder")
        self.inners_a = nn.ModuleList()  #
        self.inners_b = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(3*in_channels[i],  3*out_channels[i], ksize=1, pad=0))
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i], ksize=1, pad=0))
        self.inners_a.append(ConvBNReLU(in_channels[-1], 3*out_channels[-1], ksize=1, pad=0))  # 

        self.fuse = nn.ModuleList()
        for i in range(len(in_channels)):
            self.fuse.append(MFB(out_channels[i], out_channels[i], hidden_dims=out_channels))


    def forward(self, features, att=None): # feature[b,16,160,160], [b,24,80,80], [b,32,40,40], [b,96,20,20], [b,320,10,10]
        for idx in range(len(features) - 1, -1, -1):
            if idx == 4:
                stage_result = self.fuse[-1](self.inners_a[-1](features[-1]), idx=4)
                results = [stage_result]
            else:
                inner_lateral = features[idx]  # [1,96,20,20]
                high_up = F.interpolate(self.inners_b[idx](stage_result),size=features[idx].shape[2:],
                                              mode='bilinear',align_corners=False)
                "************"
                enhanced_fea = high_up * inner_lateral  
                cat_feat = torch.cat((high_up, enhanced_fea, inner_lateral), dim=1)  # (96,20,20)
                cat_feat = self.inners_a[idx](cat_feat)
                "************"
                stage_result = self.fuse[idx](cat_feat, idx=idx)
                results.insert(0, stage_result)
        return results

class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode='attn', args=None):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.sigmoid = nn.Sigmoid()

        if self.mode in ['original']:  #todo V1
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

        elif self.mode in ['attn']:
            self.oup = oup  # 320
            init_channels = math.ceil(oup / ratio)  # 160
            new_channels = init_channels * (ratio - 1)  # 160
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),  
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False), 
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):  # (1,320,10,10)
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))  
            x1 = self.primary_conv(x)  
            x2 = self.cheap_operation(x1)  
            out = torch.cat([x1, x2], dim=1)  
            return out[:, :self.oup, :, :] * F.interpolate(self.sigmoid(res), size=out.shape[-1], mode='nearest')
