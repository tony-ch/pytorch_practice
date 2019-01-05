#!/usr/bin/env python
# -*- coding -*-

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'VGG','VGG11_BN','VGG13','VGG13_BN','VGG16','VGG16_BN','VGG19','VGG19_BN'
]

cfgs = {
    'A':[64,   'M',128,    'M',256,256,        'M',512,512,        'M',512,512,        'M' ],#VGG11
    'B':[64,64,'M',128,128,'M',256,256,        'M',512,512,        'M',512,512,        'M' ],#VGG13
    'D':[64,64,'M',128,128,'M',256,256,256,    'M',512,512,512,    'M',512,512,512,    'M' ],#VGG16
    'E':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M' ] #VGG19
}

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

class VGG(nn.Module):
    def __init__(self, net_config, add_bn=False, num_classes=10, init_weights=True):
        super().__init__()
        assert net_config in ['A','B','D','E']
        self.features = self._make_layers(net_config, add_bn)
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096,num_classes)
        )
        if init_weights:
            self._init_weights()

    def _make_layers(self, net_cfg, add_bn):
        cfg = cfgs[net_cfg]
        layers=[]
        in_channel = 3
        for m in cfg:
            if m == 'M':
                layers.append(nn.MaxPool2d(2,stride = 2))
            else:
                layers.append(nn.Conv2d(in_channel, m, kernel_size=3, padding=1))
                if add_bn:
                    layers.append(nn.BatchNorm2d(m))
                layers.append(nn.ReLU(inplace=True))
                in_channel = m
        return nn.Sequential(*layers)

    def _init_weights(self):
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,512*7*7)
        x = self.classifier(x)
        return x


class VGG11(VGG):
    def __init__(self, num_classes=10, init_weights=True, pretrained=False):
        self.name = 'vgg11'
        super().__init__('A',False,num_classes,init_weights)
        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = model_zoo.load_url(model_urls['vgg11'])
            #将pretrained_dict里不属于model_dict的键剔除掉
            if not num_classes==1000:
                pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in {'classifier.6.bias','classifier.6.weight'}} 
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class VGG11_BN(VGG):
    def __init__(self, num_classes=10, init_weights=True, pretrained=False):
        self.name = 'vgg11_bn'
        super().__init__('A',True,num_classes,init_weights)
        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = model_zoo.load_url(model_urls['vgg11_bn'])
            #将pretrained_dict里不属于model_dict的键剔除掉
            if not num_classes==1000:
                pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in {'classifier.6.bias','classifier.6.weight'}} 
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class VGG13(VGG):
    def __init__(self, num_classes=10, init_weights=True, pretrained=False):
        self.name = 'vgg13'
        super().__init__('B',False,num_classes,init_weights)
        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = model_zoo.load_url(model_urls['vgg13'])
            #将pretrained_dict里不属于model_dict的键剔除掉
            if not num_classes==1000:
                pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in {'classifier.6.bias','classifier.6.weight'}} 
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class VGG13_BN(VGG):
    def __init__(self, num_classes=10, init_weights=True, pretrained=False):
        self.name = 'vgg13_bn'
        super().__init__('B',True,num_classes,init_weights)
        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = model_zoo.load_url(model_urls['vgg13_bn'])
            #将pretrained_dict里不属于model_dict的键剔除掉
            if not num_classes==1000:
                pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in {'classifier.6.bias','classifier.6.weight'}} 
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class VGG16(VGG):
    def __init__(self, num_classes=10, init_weights=True, pretrained=False):
        self.name = 'vgg16'
        super().__init__('D',False,num_classes,init_weights)
        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
            #将pretrained_dict里不属于model_dict的键剔除掉
            if not num_classes==1000:
                pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in {'classifier.6.bias','classifier.6.weight'}} 
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class VGG16_BN(VGG):
    def __init__(self, num_classes=10, init_weights=True, pretrained=False):
        self.name = 'vgg16_bn'
        super().__init__('B',True,num_classes,init_weights)
        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = model_zoo.load_url(model_urls['vgg16_bn'])
            #将pretrained_dict里不属于model_dict的键剔除掉
            if not num_classes==1000:
                pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in {'classifier.6.bias','classifier.6.weight'}} 
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class VGG19(VGG):
    def __init__(self, num_classes=10, init_weights=True, pretrained=False):
        self.name = 'vgg19'
        super().__init__('E',False,num_classes,init_weights)
        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = model_zoo.load_url(model_urls['vgg19'])
            #将pretrained_dict里不属于model_dict的键剔除掉
            if not num_classes==1000:
                pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in {'classifier.6.bias','classifier.6.weight'}} 
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class VGG19_BN(VGG):
    def __init__(self, num_classes=10, init_weights=True, pretrained=False):
        self.name = 'vgg19_bn'
        super().__init__('B',True,num_classes,init_weights)
        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = model_zoo.load_url(model_urls['vgg19_bn'])
            #将pretrained_dict里不属于model_dict的键剔除掉
            if not num_classes==1000:
                pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in {'classifier.6.bias','classifier.6.weight'}} 
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
