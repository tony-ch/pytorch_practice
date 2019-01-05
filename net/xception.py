#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__all__ = ['Xception']

import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, input_channel, output_channel,kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size,**kwargs)
        self.bn = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        return self.bn(self.conv(x))


class SeparableConv2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, **kwargs):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(input_channel, input_channel, kernel_size, groups=input_channel, **kwargs)
        self.pointwise_conv = nn.Conv2d(input_channel, output_channel, 1)
        self.bn = nn.BatchNorm2d(output_channel)
    def forward(self, x):
        return self.bn(self.pointwise_conv(self.depthwise_conv(x)))


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, repeat=2, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        if out_channel!=in_channel or stride!=1:
            self.skip = BasicConv2d(in_channel,out_channel,kernel_size=1,stride=stride)
        else:
            self.skip=None
        layers=[]
        mid_channel=in_channel
        if grow_first:
            layers.append(nn.ReLU())
            layers.append(SeparableConv2d(in_channel,out_channel,kernel_size=3,padding=1))
            mid_channel=out_channel
        for i in range(1,repeat):
            layers.append(nn.ReLU())
            layers.append(SeparableConv2d(mid_channel, mid_channel, kernel_size=3,padding=1))
        if not grow_first:
            layers.append(nn.ReLU())
            layers.append(SeparableConv2d(mid_channel,out_channel, kernel_size=3,padding=1))
        if not start_with_relu:
            layers=layers[1:]
        if stride!=1:
            layers.append(nn.MaxPool2d(kernel_size=3,stride=stride,padding=1))
        self.layers=nn.Sequential(*layers)

    def forward(self, x):
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip=x
        x = self.layers(x)
        return x+skip

class Xception(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super().__init__()
        self.name = 'xception'

        self.entry_flow=nn.Sequential(
            BasicConv2d(3,32,kernel_size=3,stride=2),
            nn.ReLU(),
            BasicConv2d(32,64,kernel_size=3),
            nn.ReLU(),
            Block(64,128,repeat=2,stride=2,start_with_relu=False,grow_first=True),
            Block(128,256,repeat=2,stride=2,start_with_relu=False,grow_first=True),
            Block(256,728,repeat=2,stride=2,start_with_relu=False,grow_first=True)
        )

        mid_flow=[]
        for i in range(8):
            mid_flow.append(Block(728,728,repeat=3,stride=1,start_with_relu=True,grow_first=True))
        self.mid_flow=nn.Sequential(*mid_flow)


        self.exit_flow=nn.Sequential(
            Block(728,1024,repeat=2,stride=2,start_with_relu=True,grow_first=False),
            SeparableConv2d(1024,1536,kernel_size=3),
            nn.ReLU(),
            SeparableConv2d(1536,2048,kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(2048,num_classes)


    def forward(self, x):
        x = self.entry_flow(x)
        x = self.mid_flow(x)
        x = self.exit_flow(x)
        x = self.fc(x.view(x.size(0),-1))
        return x

if __name__ == '__main__':
    net = Xception()
    print(net)
    x = torch.rand(1,3,299,299)
    print(net(x))
