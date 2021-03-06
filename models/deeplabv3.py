"""Pyramid Scene Parsing Network"""
from utils.my_seed import  seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Segment_Base import SegBaseModel, model_params, _FCNHead


__all__ = ['DeepLabV3']


class DeepLabV3(SegBaseModel):
    r"""DeepLabV3

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """

    def __init__(self, nclass, backbone='resnet50', stage='c3', aux=False, pretrained_base=True, **kwargs):
        super(DeepLabV3, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)

        self.stage = stage
        self.aux  = aux
        self.aspp = ASPP(in_channels=model_params[self.stage][backbone], num_classes=nclass)
        if self.aux:
            self.auxlayer = _FCNHead(model_params['c3'][backbone], nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)
        x = self.aspp(eval(self.stage))
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return tuple(outputs)

class ASPP(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.GroupNorm(num_channels=256, num_groups=2)

        self.conv_3x3_1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.GroupNorm(num_channels=256, num_groups=2)

        self.conv_3x3_2 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.GroupNorm(num_channels=256, num_groups=2)

        self.conv_3x3_3 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.GroupNorm(num_channels=256, num_groups=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.GroupNorm(num_channels=256, num_groups=2)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.GroupNorm(num_channels=256, num_groups=2)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))

        return out


if __name__ == '__main__':


    model = DeepLabV3(20, backbone='resnest101',stage='c4', pretrained_base= False,)
    img = torch.randn(2, 3, 480, 480)
    output = model(img)
    print(output[0].size())
