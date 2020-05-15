from utils.my_seed import  seed_everything

import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def load_pretrainedweights(model, pretrained_model):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

class VGG(nn.Module):

    def __init__(self,arc,in_channels=3,block_layers = [5,10,17,24,31] ,init_weights=False):
        super(VGG, self).__init__()
        
        self.features = make_layers(cfgs[arc], batch_norm=False)
        self.block_layers = block_layers

    def forward(self, x):
        blocks = []
        for i in self.block_layers:
            blocks.append(self.features[:i+1](x))
            
        return blocks

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg16(pretrained_model=None,  **kwargs):

    model = VGG(arc='D',in_channels=3,block_layers = [5,10,17,24,31], init_weights=False)
    if pretrained_model == 'download':
        state_dict = load_state_dict_from_url(model_urls['vgg16'],progress=True)
        model.load_state_dict(state_dict)
    elif pretrained_model is not None:
        model = load_pretrainedweights(model, pretrained_model)
    return model



def vgg19(pretrained_model=None,  **kwargs):

    
    model = VGG(arc='E',in_channels=3,block_layers = [5,10,19,28,37], init_weights=False)
    if pretrained_model == 'download':
        state_dict = load_state_dict_from_url(model_urls['vgg19'],progress=True)
        model.load_state_dict(state_dict)
    elif pretrained_model is not None:
        model = load_pretrainedweights(model, pretrained_model)
    return model


def print_weight(model):
    model_dict = model.state_dict()
    for k,v in model_dict.items():
        print("layer name:",k,v.size())
        print(v[0,0,:3,:3])
        exit()

if __name__ == '__main__':

    x = torch.randn(1,3,512,512)
    model = vgg16(pretrained_model = "/home/zf/.torch/models/vgg16-397923af.pth")
    # model = vgg19(pretrained_model = "/home/zf/.torch/models/vgg19-dcbb9e9d.pth")
    outputs = model(x)
    for output in outputs:
        print(output.size())
    print_weight(model)

