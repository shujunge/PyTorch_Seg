import torch
import torch.nn as nn
from backbone.resnetv1b import resnet50_v1b, resnet101_v1b, resnet152_v1b
from backbone.ResNest.resnest import resnest50, resnest101
from backbone.resnet import resnet101, resnet50
from backbone.EfficientNet import EfficientNet_B4


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


class CBNet(nn.Module):

    def __init__(self, backbone1='resnest50', backbone2='resnest101', pretrained_base=True, **kwargs):
        super(CBNet, self).__init__()

        #  backbone in ['resnest50', 'resnest101']
        self.backbone1 = eval(backbone1)(pretrained=pretrained_base, **kwargs)
        self.backbone2 = eval(backbone2)(pretrained=pretrained_base, **kwargs)
        self.x_1_2 = _ConvGNReLU(64, 128, kernel_size=1, stride=1, padding=0,)
        self.x_fuse_1_2 = _ConvGNReLU(128, 128, kernel_size=3, stride=1, padding=1,)
        self.c1_1_2 = _ConvGNReLU(256, 256, kernel_size=1, stride=1, padding=0, )
        self.c1_fuse_1_2 = _ConvGNReLU(256, 256, kernel_size=3, stride=1, padding=1, )
        self.c2_1_2 = _ConvGNReLU(512, 512, kernel_size=1, stride=1, padding=0, )
        self.c2_fuse_1_2 = _ConvGNReLU(512, 512, kernel_size=3, stride=1, padding=1, )
        self.c3_1_2 = _ConvGNReLU(1024, 1024, kernel_size=1, stride=1, padding=0, )
        self.c3_fuse_1_2 = _ConvGNReLU(1024, 1024, kernel_size=3, stride=1, padding=1, )
        self.c4_1_2 = _ConvGNReLU(2048, 2048, kernel_size=1, stride=1, padding=0, )
        self.c4_fuse_1_2 = _ConvGNReLU(2048, 2048, kernel_size=3, stride=1, padding=1, )

    def get_base_stage(self,x):
        x = self.backbone1.conv1(x)
        x = self.backbone1.bn1(x)
        x = self.backbone1.relu(x)
        x = self.backbone1.maxpool(x)
        c1 = self.backbone1.layer1(x)
        c2 = self.backbone1.layer2(c1)
        c3 = self.backbone1.layer3(c2)
        c4 = self.backbone1.layer4(c3)

        return x, c1, c2, c3, c4

    def forward(self, x):
        """forwarding pre-trained network"""
        x1, base_c1, base_c2, base_c3, base_c4 = self.get_base_stage(x)
        x = self.backbone2.conv1(x)
        x = self.backbone2.bn1(x)
        x = self.backbone2.relu(x)
        x = self.backbone2.maxpool(x)
        x = self.x_fuse_1_2(self.x_1_2(x1) + x)
        c1 = self.backbone2.layer1(x)
        c1 = self.c1_fuse_1_2(self.c1_fuse_1_2(base_c1) + c1)
        c2 = self.backbone2.layer2(c1)
        c2 = self.c2_fuse_1_2(self.c2_1_2(base_c2) + c2)
        c3 = self.backbone2.layer3(c2)
        c3 = self.c3_fuse_1_2(self.c3_1_2(base_c3) + c3)
        c4 = self.backbone2.layer4(c3)
        c4 = self.c4_fuse_1_2(self.c4_1_2(base_c4) + c4)

        return c1, c2, c3, c4

if __name__=="__main__":
    model = CBNet(backbone1='resnest50', backbone2='resnest101', pretrained_base=False)
    x = torch.randn((2,3,224,224))
    outputs = model(x)
    for output in outputs:
        print(output.size())


"""
#resnest50
        # [2, 64, 56, 56])
        # torch.Size([2, 256, 56, 56])
        # torch.Size([2, 512, 28, 28])
        # torch.Size([2, 1024, 14, 14])
        # torch.Size([2, 2048, 7, 7])
#resnest101
        # torch.Size([2, 128, 56, 56])
        # torch.Size([2, 256, 56, 56])
        # torch.Size([2, 512, 28, 28])
        # torch.Size([2, 1024, 14, 14])
        # torch.Size([2, 2048, 7, 7])
"""