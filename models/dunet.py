"""Decoders Matter for Semantic Segmentation"""
from utils.my_seed import  seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Segment_Base import SegBaseModel


__all__ = ['DUNet']


# The model may be wrong because lots of details missing in paper.
class DUNet(SegBaseModel):
    """Decoders Matter for Semantic Segmentation

    Reference:
        Zhi Tian, Tong He, Chunhua Shen, and Youliang Yan.
        "Decoders Matter for Semantic Segmentation:
        Data-Dependent Decoding Enables Flexible Feature Aggregation." CVPR, 2019
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, pretrained_base=True, **kwargs):
        super(DUNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _DUHead(2144, **kwargs)
        self.dupsample = DUpsampling(256, nclass, scale_factor=8, **kwargs)

        self.__setattr__('exclusive',
                         ['dupsample', 'head', 'auxlayer', 'aux_dupsample'] if aux else ['dupsample', 'head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c2, c3, c4)
        x = self.dupsample(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        #
        # if self.aux:
        #     auxout = self.auxlayer(c3)
        #     auxout = self.aux_dupsample(auxout)
        #     outputs.append(auxout)
        return tuple(outputs)


class FeatureFused(nn.Module):
    """Module for fused features"""

    def __init__(self, inter_channels=48, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FeatureFused, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1024, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
        )

    def forward(self, c2, c3, c4):
        size = c4.size()[2:]
        c2 = self.conv2(F.interpolate(c2, size, mode='bilinear', align_corners=True))
        c3 = self.conv3(F.interpolate(c3, size, mode='bilinear', align_corners=True))
        fused_feature = torch.cat([c4, c3, c2], dim=1)
        return fused_feature


class _DUHead(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DUHead, self).__init__()
        self.fuse = FeatureFused(norm_layer=norm_layer, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(True)
        )

    def forward(self, c2, c3, c4):
        fused_feature = self.fuse(c2, c3, c4)
        out = self.block(fused_feature)
        return out


class DUpsampling(nn.Module):
    """DUsampling module"""

    def __init__(self, in_channels, out_channels, scale_factor=2, **kwargs):
        super(DUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_w = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, 1, bias=False)

    def forward(self, x):
        x = self.conv_w(x)
        n, c, h, w = x.size()

        # N, C, H, W --> N, W, H, C
        x = x.permute(0, 3, 2, 1).contiguous()

        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, h * self.scale_factor, c // self.scale_factor)

        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()

        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, h * self.scale_factor, w * self.scale_factor, c // (self.scale_factor * self.scale_factor))

        # N, H * scale, W * scale, C // (scale ** 2) -- > N, C // (scale ** 2), H * scale, W * scale
        x = x.permute(0, 3, 1, 2)

        return x



if __name__ == '__main__':
    img = torch.randn(2, 3, 256, 256)
    model = DUNet(20, backbone='resnest101', pretrained_base=False)
    outputs = model(img)
    print(outputs.size())
