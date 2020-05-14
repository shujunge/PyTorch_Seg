from backbone.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s
from backbone.ResNest.resnest import resnest50, resnest101
from backbone.resnet import resnet101, resnet50
from backbone.EfficientNet import EfficientNet_B4
import torch
import torch.nn as nn


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50',  pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated =  True
        self.aux = aux
        self.nclass = nclass
        self.backbone = backbone

        models_name = {}
        models_name['resnet50'] = "/home/zfw/.torch/models/resnet50-19c8e357.pth"
        models_name['resnet101'] = "/home/zfw/.torch/models/resnet101-5d3b4d8f.pth"
        # models['xception39'] = "/home/zfw/.torch/models/xception-43020ad28.pth"
        models_name['EfficientNet_B4'] = "/home/zfw/.torch/models/efficientnet-b4-6ed6700e.pth"

        if backbone in[ 'resnet50_v1s','resnet101_v1s', 'resnet152_v1s']:
            self.pretrained = eval(backbone)(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone in list(models_name.keys()):
            self.pretrained = eval(backbone)(in_channels=3, pretrained_model=models_name[backbone], **kwargs) #
        elif backbone in ['resnest50', 'resnest101']:
            self.pretrained = eval(backbone)(pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))



    def base_forward(self, x):
        """forwarding pre-trained network"""

        if self.backbone == "EfficientNet_B4":
            _, c1, c2, c3, c4 = self.pretrained(x)
            return c1, c2, c3, c4
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)

            return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class _ConvGNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, **kwargs):
        super(_ConvGNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.gn = nn.GroupNorm(num_channels=out_channels,num_groups=2)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class _ConvBNPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x
