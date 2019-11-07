"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Base Model for Semantic Segmentation"""

from backbone.resnet import resnet50, resnet101
from backbone.vgg import vgg16
from backbone.EfficientNet import EfficientNet_B4

class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', pretrained_base = True, **kwargs):
        super(SegBaseModel, self).__init__()

        self.aux = aux
        self.nclass = nclass
        self.backbone = backbone
        models={}
        models['resnet18'] = "/home/zfw/.torch/models/resnet18-5c106cde.pth"
        models['resnet34'] = "/home/zfw/.torch/models/resnet34-333f7ec4.pth"
        models['resnet50'] = "/home/zfw/.torch/models/resnet50-19c8e357.pth"
        models['resnet101'] = "/home/zfw/.torch/models/resnet101-5d3b4d8f.pth"
        models['vgg16'] = "/home/zfw/.torch/models/vgg16-00b39a1b.pth"
        models['xception39'] = "/home/zfw/.torch/models/xception-43020ad28.pth"
        models['EfficientNet_B4']=None
        if pretrained_base:
            self.pretrained = eval(backbone)(pretrained_model=models[backbone] , **kwargs)
        else:
            self.pretrained = eval(backbone)(pretrained_model=None, **kwargs)


    def base_forward(self, x):
        """forwarding pre-trained network"""
        if self.backbone[:6]=="resnet":
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)#[bn, 64, 120, 120]
            c1 = self.pretrained.layer1(x) #[bn, 256, 120, 120]
            c2 = self.pretrained.layer2(c1)#[bn, 512, 60, 60]
            c3 = self.pretrained.layer3(c2)#[bn, 1024, 30, 30]
            c4 = self.pretrained.layer4(c3) #[bn, 2048, 15, 15]
            return c1, c2, c3, c4
        elif self.backbone == "vgg16":
            x = self.pretrained(x)
            return x
        else:
            x = self.pretrained(x)
            return x

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)



def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )


class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class _PSPHead(nn.Module):
    def __init__(self, nclass,backbone, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.backbone = backbone
        if  self.backbone in ['resnet50','resnet101']:
            self.psp = _PyramidPooling(2048, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.block = nn.Sequential(
                nn.Conv2d(4096, 512, 3, padding=1, bias=False),
                norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(512, nclass, 1)
            )
        elif self.backbone =='vgg16':
            self.psp = _PyramidPooling(512, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.block = nn.Sequential(
                nn.Conv2d(1024, 512, 3, padding=1, bias=False),
                norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(512, nclass, 1)
            )
        elif self.backbone =='EfficientNet_B4':
            self.psp = _PyramidPooling(1792, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.block = nn.Sequential(
                nn.Conv2d(1792*2, 512, 3, padding=1, bias=False),
                norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(512, nclass, 1)
            )

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)


class PSPNet(SegBaseModel):
    r"""Pyramid Scene Parsing Network

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
        Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia.
        "Pyramid scene parsing network." *CVPR*, 2017
    """

    def __init__(self, nclass, aux=False, backbone='resnet50',  pretrained_base=True, **kwargs):
        super(PSPNet, self).__init__(nclass, aux,backbone,  pretrained_base=pretrained_base, **kwargs)
        self.head = _PSPHead(nclass,backbone, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        x = self.base_forward(x)
        c4 =x[-1]
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x

def print_weight(model):
    model_dict = model.state_dict()
    for k,v in model_dict.items():
        print("layer name:",k,v.size())
        print(v[0,0,:3,:3])
        exit()

if __name__ == '__main__':

    model = PSPNet(150, backbone='vgg16', pretrained_base=False)
    img = torch.randn(2, 3, 480, 480)
    output = model(img)
    print("output size:",output.size())
    # print_weight(model)
    from torchsummary import summary
    summary(model,(3,512,512))

