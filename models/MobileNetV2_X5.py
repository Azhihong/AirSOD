from torch import nn
import torch.nn.functional as F
import math
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from torchvision.models.utils import load_state_dict_from_url # torchvision 0.4+
except ModuleNotFoundError:
    try:
        from torch.hub import load_state_dict_from_url # torch 1.x
    except ModuleNotFoundError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url # torch 0.4.1


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2
        if dilation != 1:
            padding = dilation
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, dilation=dilation),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
"***************CCA start**********************"
def INF(B, H, W):
    # return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):  # 输入维度64
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # [2,64,5,6]\[1,64,40,40]
        m_batchsize, _, height, width = x.size()  # batch, height, width
        proj_query = self.query_conv(x)  # [2,8,5,6]\[1,8,40,40] todo 获得Q
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,1)  # [40,40,8]
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)  # [40,40,8]

        proj_key = self.key_conv(x)  # [1,8,40,40] todo 获得K
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)  # [40,8,40]
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)  # [40,8,40]

        proj_value = self.value_conv(x)  # [1,64,40,40] todo 获得V
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)  # [40,64,40]
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)  # [40,64,40]
        # [12,5,8]*[12,8,5]=[12,5,5],->view->[2,5,6,5]
        # energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,height,height).permute(0,2,1,3)  # (1,40,40,40)
        energy_H = torch.bmm(proj_query_H, proj_key_H).view(m_batchsize, width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width,width)  # [10,6,8]*[10,8,6]=[10,6,6],->view->[2,5,6,6]  # (1,40,40,40)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))  # (1,40,40,80), 这个拼接操作只是为了做softmax

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,height)  # [40,40,40]
        # todo Aggregation
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width,width)  # [40,40,40]
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3,1)  # [1,64,40,40]
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1,3)  # [1,64,40,40]
        return self.gamma * (out_H + out_W)  #  + x
"*************CCA end*************"

"*************Tok-Mlp start*********************"
class fea_split(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride,padding=padding)
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

    def forward(self, x):  # (b,64,40,40)/(b,128,20,20)
        x = self.proj(x)  # 输出x(b,128,20,20)/(b,320,10,10)

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
        x = self.dwconv(x)  # x:(b,160,32,32)
        return x

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1,padding=0, bias=False)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()  # 激活层：nn.GELU
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 =nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1,padding=0, bias=False)

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
        B, C, H, W = x.shape

        # todo Shifted MLP (Width)
        # xn = x.transpose(1, 2).view(B, C, H, W).contiguous()  # xn[b,160,32,32]/[b,256,16,16]
        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)   # xn[b,160,36,36]/[b,256,20,20]
        xs = torch.chunk(xn, self.shift_size, 1)  # xs[(b,32,36,36),(b,32,36,36),(b,32,36,36),(b,32,36,36),(b,32,36,36)]
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]  # shift=-2,-1,0,1,2, x_shift:[tensor(b,32,36,36),tensor(b,32,36,36),tensor(b,32,36,36),tensor(b,32,36,36),tensor(b,32,36,36),]
        x_cat = torch.cat(x_shift, 1)  # (b,160,36,36)  todo 上句重点:使得相邻的patch之间更好地通讯
        x_cat = torch.narrow(x_cat, 2, self.pad, H)  # x_cat:(b,160,32,36)  torch.narrow(input, dim, start, length)
        x_s = torch.narrow(x_cat, 3, self.pad, W)  # x_s:(b,160,32,32)

        # todo project, 对应下面的reproject, 不过这个project在文中图2没有画出来
        x = self.fc1(x_s)  # x:(b,1024,160), 全连接层，通道数不变
        x = self.dwconv(x)  # x:(b,1024,160)
        x = self.act(x)  # 激活层 todo GELU

        # todo Shifted MLP (Height)
        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)  # xn:(b,160,36,36)
        xs = torch.chunk(xn, self.shift_size, 1) # xs名字中的s应该指split, # xs[(b,32,36,36),(b,32,36,36),(b,32,36,36),(b,32,36,36),(b,32,36,36)]
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]  # shift=-2,-1,0,1,2, x_shift:[tensor(b,32,36,36),tensor(b,32,36,36),tensor(b,32,36,36),tensor(b,32,36,36),tensor(b,32,36,36),]
        x_cat = torch.cat(x_shift, 1)  # x_cat:(b,160,36,36)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)  # x_cat:(b,160,32,36)
        x_s = torch.narrow(x_cat, 3, self.pad, W)  # x_s:(b,160,32,32)

        # todo Reproject
        x = self.fc2(x_s)  # x:[b,1024,160]
        return x

class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = nn.BatchNorm2d(dim)
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

    def forward(self, x):  # x(b,48,20,20)
        # x = x + self.drop_path(self.mlp(self.norm2(x), H, W))  # todo 残差连接
        x = self.drop_path(self.mlp(x))
        return x
"*************Tok-Mlp end***********************"

class MobileNetV2(nn.Module):
    def __init__(self, pretrained=None, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        print("Using Hybrid Architecture MobileNetV2_X5")
        block = InvertedResidual
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 2, 1],
        ]  #[6, 96, 3, 1, 1],[6, 160, 3, 2, 1], [6, 320, 1, 1, 1],

        # building first layer
        input_channel = int(input_channel * width_mult)  # 第一层卷积32
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s, d in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                dilation = d if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, dilation=d))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        "************AMC start***************"
        drop_path_rate = 0.
        depths = [1, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        embed_dims = [32, 64, 96, 160, 320]
        num_heads = [1, 2, 4, 8]
        qkv_bias = False
        drop_rate = 0.
        attn_drop_rate = 0.
        sr_ratios = [8, 4, 2, 1]
        qk_scale = None

        self.cca2 = CrissCrossAttention(in_dim=embed_dims[2]//2)
        self.conv2 = ConvBNReLU(in_planes=embed_dims[2], out_planes=embed_dims[2], kernel_size=1)
        # todo 相比于V4,V5的变化是把3x3的卷积核改成1x1的卷积
        self.fea_split2 = fea_split(in_chans=embed_dims[1], embed_dim=embed_dims[2],kernel_size=1, stride=1)
        self.block2 = nn.ModuleList([
            shiftedBlock(
                dim=embed_dims[2]//2, num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],sr_ratio=sr_ratios[0])])
        self.norm2 =nn.BatchNorm2d(embed_dims[2]//2)
        '---------'
        self.cca3 = CrissCrossAttention(in_dim=embed_dims[3]//2)
        self.conv3 = ConvBNReLU(in_planes=embed_dims[3], out_planes=embed_dims[3], kernel_size=1)
        self.fea_split3 = fea_split(in_chans=embed_dims[2], embed_dim=embed_dims[3],kernel_size=1, stride=2)
        self.block3 = nn.ModuleList([
            shiftedBlock(
                dim=embed_dims[3]//2, num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],sr_ratio=sr_ratios[0])
        ])  # todo 为什么用nn.ModuleList呢，因为可拓展多层shiftedBlock,这里只用了一个层。
        self.norm3 = nn.BatchNorm2d(embed_dims[3]//2)
        '---------'
        self.cca4 = CrissCrossAttention(in_dim=embed_dims[4]//2)
        self.conv4 = ConvBNReLU(in_planes=embed_dims[4], out_planes=embed_dims[4], kernel_size=1)
        self.fea_split4 = fea_split(in_chans=embed_dims[3], embed_dim=embed_dims[4], kernel_size=1, stride=1)
        self.block4 = nn.ModuleList([
            shiftedBlock(
                dim=embed_dims[4]//2, num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],sr_ratio=sr_ratios[0])])
        self.norm4 = nn.BatchNorm2d(embed_dims[4]//2)
        "************AMC end***************"

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # [1,3,320,320]
        res = []
        for idx, m in enumerate(self.features):
            x = m(x)
            if idx in [1,3,6,10]:   # ,13,17
                res.append(x)
        "**************Attention-MLP Combined block Start***************"
        out = res[-1]  # out [b,64,20,20]
        x, x1, x2 = self.fea_split2(out)  # x[b,96,20,20] out[b,400,48], x2[b,48,20,20], H:20, W:20
        x2 = self.cca2(x2)  # 输出x2[b,48,20,20]
        for i, blk in enumerate(self.block2):
            x1 = blk(x1)  # out[b,400,48]
        x1 = self.norm2(x1)  # out[b,400,96]
        out = self.conv2(torch.cat((x1, x2), dim=1)) + x
        res[-1] = out

        x, x1, x2 = self.fea_split3(out)  # out[b,400,96], H:20, W:20
        x2 = self.cca3(x2)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1)  # out[b,400,96]
        x1 = self.norm3(x1)  # out[b,400,96]
        out = self.conv3(torch.cat((x1,x2), dim=1)) + x

        x, x1, x2 = self.fea_split4(out)  # out[b,100,320], H:16, W:16
        x2 = self.cca4(x2)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1)  # out[b,100,320]
        x1 = self.norm4(x1)  # out[b,100,320]
        out = self.conv4(torch.cat((x1, x2), dim=1)) + x
        res.append(out)
        "**************Attention-MLP Combined block End***************"
        return res


def mobilenet_v2(pretrained=True, progress=True, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)  # 从url中下载预训练模型
        print("loading imagenet pretrained mobilenetv2")
        model.load_state_dict(state_dict, strict=False)  # 把下载的预训练模型加载进来
        print("loaded imagenet pretrained mobilenetv2")
    return model

