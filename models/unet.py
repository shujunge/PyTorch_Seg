""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.resnet import resnet50, resnet34,resnet18,resnet101
from backbone.EfficientNet import EfficientNet_B4
from backbone.vgg import vgg16

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True, backbone= 'resnet50', pretrained_base =False, usehypercolumns =False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.usehypercolumns  = usehypercolumns

        models = {}
        models['resnet18'] = "/home/zfw/.torch/models/resnet18-5c106cde.pth"
        models['resnet34'] = "/home/zfw/.torch/models/resnet34-333f7ec4.pth"
        models['resnet50'] = "/home/zfw/.torch/models/resnet50-19c8e357.pth"
        models['resnet101'] = "/home/zfw/.torch/models/resnet101-5d3b4d8f.pth"
        models['vgg16'] = "/home/zfw/.torch/models/vgg16-00b39a1b.pth"
        models['xception39'] = "/home/zfw/.torch/models/xception-43020ad28.pth"
        models['EfficientNet_B4'] =  "/home/zfw/.torch/models/efficientnet-b4-6ed6700e.pth"

        if pretrained_base:
            self.pretrained = eval(backbone)(in_channels=self.in_channels, pretrained_model=models[backbone])
        else:
            self.pretrained = eval(backbone)(in_channels=self.in_channels, pretrained_model=None)

        self.params = {'EfficientNet_B4':[1952, 312, 160, 88, 64], 'vgg16':[1024,768, 384,192,64],
                       'resnet34':[768, 384, 192,128, 64], 'resnet50':[3072,768,384, 128, 64],'resnet101':[3072,768,384,128,64]}

        n1, n2, n3, n4, n5 = self.params[backbone]

        self.up1 = Up(n1, 256, bilinear)
        self.up2 = Up(n2, 128, bilinear)
        self.up3 = Up(n3, 64, bilinear)
        self.up4 = Up(n4, 64, bilinear)
        self.outc = OutConv(n5, n_classes)

        self.last = nn.Sequential(nn.Conv2d(449, 32, 3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 1, 3, padding=1)
                                  )

    def forward(self, x):
        img_size = x.size()[-2:]
        x1, x2, x3, x4, x5 = self.pretrained(x)
        up1 = self.up1(x5, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        up4 = self.up4(up3, x1)
        logits = self.outc(up4)

        if self.usehypercolumns:
            temp = torch.cat([logits, F.interpolate(up1, img_size, mode='bilinear', align_corners=True),
                              F.interpolate(up2, img_size, mode='bilinear', align_corners=True),
                              F.interpolate(up3, img_size, mode='bilinear', align_corners=True)], dim=1)

            logits = self.last(temp)
        return logits

if __name__ == '__main__':

    x = torch.randn(2, 6, 128, 128)
    model = UNet(in_channels=6, n_classes=1, backbone='resnet34', usehypercolumns=True)
    out = model(x)
    print(out.size())
    model = UNet(in_channels=6, n_classes=1, backbone='resnet50', usehypercolumns=True)
    out = model(x)
    print(out.size())
    model = UNet(in_channels=6, n_classes=1, backbone='EfficientNet_B4', usehypercolumns=True)
    out = model(x)
    print(out.size())
    # from torchsummary import summary
    # summary(model, (6, 128, 128))


"""
'EfficientNet_B4'
================================================================
Total params: 23,362,701
Trainable params: 23,362,701
Non-trainable params: 0
----------------------------------------------------------------
'resnet34'
================================================================
Total params: 24,495,237
Trainable params: 24,495,237
Non-trainable params: 0
----------------------------------------------------------------
'resnet50'
================================================================
Total params: 32,579,973
Trainable params: 32,579,973
Non-trainable params: 0
----------------------------------------------------------------

'vgg16'
================================================================
Total params: 32,683,525
Trainable params: 32,683,525
Non-trainable params: 0
----------------------------------------------------------------

"""
