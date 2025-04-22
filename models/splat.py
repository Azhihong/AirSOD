"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d
from torch.nn import BatchNorm2d as get_norm


# from detectron2.layers import (
#     Conv2d,
#     get_norm,
# )

__all__ = ['SplAtConv2d', 'SplAtConv2d_dcn']

class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
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
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm is not None
        if self.use_bn:
            self.bn0 = get_norm(norm, channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = get_norm(norm, inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)  # radix:2, groups:1
        "***********multi-scale start***********"
        hidden_dim=160
        dilation=[1,2,3]
        self.hidden_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[0], groups=hidden_dim,
                                      dilation=dilation[0])
        self.hidden_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[1], groups=hidden_dim,
                                      dilation=dilation[1])
        self.hidden_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[2], groups=hidden_dim,
                                      dilation=dilation[2])
        self.hidden_bnact1 = nn.Sequential(nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.hidden_bnact2 = nn.Sequential(nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.hidden_bnact3 = nn.Sequential(nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        "***********multi-scale end*************"

    def forward(self, x):  # (1,160,10,10)
        x = self.conv(x)  # 输出x(1,320,10,10)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)  # (1,320,10,10)

        batch, rchannel = x.shape[:2]  # batch:1, rchannel:320
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)  # splited(tuple:2)([1,160,10,10],[1,160,10,10])
            "********multi-scale start**********"
            m1 = self.hidden_conv1(splited[0])
            m1 = self.hidden_bnact1(m1)
            m2 = self.hidden_conv2(splited[1])
            m2 = self.hidden_bnact2(m2)
            # m3 = self.hidden_conv3(splited[2])
            # m3 = self.hidden_bnact3(m3)
            m = (m1,m2)  # ,m3
            "********multi-scale end************"
            gap = sum(splited)  # gap[1,160,10,10]
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)  # gap[1,160,1,1]
        gap = self.fc1(gap)  # fc1是用1x1的卷积来实现，gap[1,80,1,1]
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)  # gap[1,80,1,1]
        atten = self.fc2(gap)  # fc2也是用1x1的卷积来实现  # atten[1,320,1,1]
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)  # atten[1,320,1,1] todo 通道注意力

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)  # attens(tuple:2)([1,160,10,10],[1,160,10,10])
            out = sum([att*split for (att, split) in zip(attens, splited)])  # out[1,160,10,10]
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


class SplAtConv2d_dcn(Module):
    """Split-Attention Conv2d with dcn
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm=None,
                 dropblock_prob=0.0,
                 deform_conv_op=None,
                 deformable_groups=1,
                 deform_modulated=False,
                 **kwargs):
        super(SplAtConv2d_dcn, self).__init__()
        self.deform_modulated = deform_modulated

        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = deform_conv_op(in_channels, channels*radix, kernel_size, stride, padding[0], dilation,
                               groups=groups*radix, bias=bias, deformable_groups=deformable_groups, **kwargs)
        self.use_bn = norm is not None
        if self.use_bn:
            self.bn0 = get_norm(norm, channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = get_norm(norm, inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x, offset_input):

        if self.deform_modulated:
            offset_x, offset_y, mask = torch.chunk(offset_input, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            x = self.conv(x, offset, mask)
        else:
            x = self.conv(x, offset_input)

        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SplAtConv2d(in_channels=160, channels=160, kernel_size=1)  # 十字交叉注意力模块不改变输入输出的大小
    model.to(device)
    x = torch.randn(1, 160, 10, 10).to(device)  # 输入[b,64,5,6]
    out = model(x)  # out[b,64,5,6]
    print(out.shape)