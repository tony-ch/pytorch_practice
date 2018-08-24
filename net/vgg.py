#!/usr/bin/env python
# -*- coding -*-

import torch.nn as nn

__all__ = [
    'VGG','VGG11_BN','VGG13','VGG13_BN','VGG16','VGG16_BN','VGG19','VGG19_BN'
]

cfgs = {
    'A':[64,   'M',128,    'M',256,256,        'M',512,512,        'M',512,512,        'M' ],#VGG11
    'B':[64,64,'M',128,128,'M',256,256,        'M',512,512,        'M',512,512,        'M' ],#VGG13
    'D':[64,64,'M',128,128,'M',256,256,256,    'M',512,512,512,    'M',512,512,512,    'M' ],#VGG16
    'E':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M' ] #VGG19
}


class VGG(nn.Module):
    def __init__(self, net_config, add_bn=False, num_classes=10, init_weights=True):
        super().__init__()
        assert net_config in ['A','B','D','E']
        self.convs = self._make_layers(net_config, add_bn)
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
        x = self.convs(x)
        x = x.view(-1,512*7*7)
        x = self.classifier(x)
        return x


class VGG11(VGG):
    def __init__(self, num_classes=10, init_weights=True):
        self.name = 'vgg11'
        super().__init__('A',False,num_classes,init_weights)

class VGG11_BN(VGG):
    def __init__(self, num_classes=10, init_weights=True):
        self.name = 'vgg11_bn'
        super().__init__('A',True,num_classes,init_weights)

class VGG13(VGG):
    def __init__(self, num_classes=10, init_weights=True):
        self.name = 'vgg13'
        super().__init__('B',False,num_classes,init_weights)

class VGG13_BN(VGG):
    def __init__(self, num_classes=10, init_weights=True):
        self.name = 'vgg13_bn'
        super().__init__('B',True,num_classes,init_weights)

class VGG16(VGG):
    def __init__(self, num_classes=10, init_weights=True):
        self.name = 'vgg16'
        super().__init__('D',False,num_classes,init_weights)

class VGG16_BN(VGG):
    def __init__(self, num_classes=10, init_weights=True):
        self.name = 'vgg16_bn'
        super().__init__('B',True,num_classes,init_weights)

class VGG19(VGG):
    def __init__(self, num_classes=10, init_weights=True):
        self.name = 'vgg19'
        super().__init__('E',False,num_classes,init_weights)

class VGG19_BN(VGG):
    def __init__(self, num_classes=10, init_weights=True):
        self.name = 'vgg19_bn'
        super().__init__('B',True,num_classes,init_weights)
