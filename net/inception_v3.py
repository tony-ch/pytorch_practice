#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Inception_v3']

class Inception_v3(nn.Module):
    def __init__(self, input_channel=3, output_channel=10):
        super().__init__()
        self.name = 'inception_v3'
        self.conv1_3x3 = BasicConv2d(input_channel, 32, kernel_size=3, stride=2)

        self.conv2_3x3 = BasicConv2d(32, 32, kernel_size=3)

        self.conv3_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        # pool
        self.conv4a_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.conv4b_3x3 = BasicConv2d(80, 192, kernel_size=3)
        # pool
        self.mixed_5a = InceptionA(192, pool_features=32)
        self.mixed_5b = InceptionA(256, pool_features=64)
        self.mixed_5c = InceptionA(288, pool_features=64)

        self.mixed_6a = InceptionB1(288)
        self.mixed_6b = InceptionB2(768, channels_7x7=128)
        self.mixed_6c = InceptionB2(768, channels_7x7=160)
        self.mixed_6d = InceptionB2(768, channels_7x7=160)
        self.mixed_6e = InceptionB2(768, channels_7x7=192)

        self.mixed_7a = InceptionC1(768)
        self.mixed_7b = InceptionC2(1280)
        self.mixed_7c = InceptionC2(2048)

        #pool
        self.fc = nn.Linear(2048, output_channel)
    def forward(self, x):
        # 299x299x3
        conv1 = self.conv1_3x3(x)

        # 149x149x32
        conv2 = self.conv2_3x3(conv1)

       # 147x147x32
        conv3 = self.conv3_3x3(conv2)
        # 147x147x64
        conv3 = F.max_pool2d(conv3, kernel_size=3, stride=2)

        # 73x73x64
        conv4 = self.conv4a_1x1(conv3)
        # 73x73x80
        conv4 = self.conv4b_3x3(conv4)
        # 71x71x192
        conv4 = F.max_pool2d(conv4, kernel_size=3, stride=2)

        # 35x35x192
        mixed5 = self.mixed_5a(conv4)
        # 35x35x256
        mixed5 = self.mixed_5b(mixed5)
        # 35x35x288
        mixed5 = self.mixed_5c(mixed5)

        mixed6 = self.mixed_6a(mixed5)
        # 17x17x768
        mixed6 = self.mixed_6b(mixed6)
        mixed6 = self.mixed_6c(mixed6)
        mixed6 = self.mixed_6d(mixed6)
        mixed6 = self.mixed_6e(mixed6)

        mixed7 = self.mixed_7a(mixed6)
        # 8x8x1280
        mixed7 = self.mixed_7b(mixed7)
        # 8x8x2048
        mixed7 = self.mixed_7c(mixed7)

        # 8x8x2048
        features = F.avg_pool2d(mixed7, kernel_size=8)
        # 1x1x2048
        features = F.dropout(features, training=self.training)
        # 2048
        features = features.view(features.size(0),-1)

        res = self.fc(features)
        return res

class InceptionA(nn.Module):
    def __init__(self, input_channel,  pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channel, 64, kernel_size=1)

        self.branch5x5_1a = BasicConv2d(input_channel, 48, kernel_size=1)
        self.branch5x5_1b = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch5x5_2a = BasicConv2d(input_channel, 64, kernel_size=1)
        self.branch5x5_2b = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch5x5_2c = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(input_channel, pool_features, kernel_size=1)
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5_1 = self.branch5x5_1a(x)
        branch5x5_1 = self.branch5x5_1b(branch5x5_1)

        branch5x5_2 = self.branch5x5_2a(x)
        branch5x5_2 = self.branch5x5_2b(branch5x5_2)
        branch5x5_2 = self.branch5x5_2c(branch5x5_2)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(x)
        output = [branch1x1, branch5x5_1, branch5x5_2, branch_pool]
        return torch.cat(output, 1)


class InceptionB1(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.branch3x3 = BasicConv2d(input_channel, 384, kernel_size=3, stride=2)

        self.branch5x5a = BasicConv2d(input_channel, 64, kernel_size=1)
        self.branch5x5b = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch5x5c = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch5x5 = self.branch5x5a(x)
        branch5x5 = self.branch5x5b(branch5x5)
        branch5x5 = self.branch5x5c(branch5x5)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        output = [branch3x3, branch5x5, branch_pool]
        return torch.cat(output, 1)

class InceptionB2(nn.Module):
    def __init__(self, input_channel, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channel, 192, kernel_size =1)

        self.branch7x7_1a = BasicConv2d(input_channel, channels_7x7, kernel_size=1)
        self.branch7x7_1b = BasicConv2d(channels_7x7, channels_7x7, kernel_size = (1,7), padding=(0,3))
        self.branch7x7_1c = BasicConv2d(channels_7x7, 192, kernel_size=(7,1), padding=(3,0))

        self.branch7x7_2a = BasicConv2d(input_channel, channels_7x7, kernel_size=1)
        self.branch7x7_2b = BasicConv2d(channels_7x7, channels_7x7, kernel_size = (7,1), padding =(3,0))
        self.branch7x7_2c = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1,7), padding=(0,3))
        self.branch7x7_2d = BasicConv2d(channels_7x7, channels_7x7, kernel_size = (7,1), padding =(3,0))
        self.branch7x7_2e = BasicConv2d(channels_7x7, 192, kernel_size=(1,7), padding=(0,3))

        self.branch_pool = BasicConv2d(input_channel, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7_1 = self.branch7x7_1a(x)
        branch7x7_1 = self.branch7x7_1b(branch7x7_1)
        branch7x7_1 = self.branch7x7_1c(branch7x7_1)

        branch7x7_2 = self.branch7x7_2a(x)
        branch7x7_2 = self.branch7x7_2b(branch7x7_2)
        branch7x7_2 = self.branch7x7_2c(branch7x7_2)
        branch7x7_2 = self.branch7x7_2d(branch7x7_2)
        branch7x7_2 = self.branch7x7_2e(branch7x7_2)

        branch_pool = self.branch_pool(x)

        output = [branch1x1, branch7x7_1, branch7x7_2, branch_pool]
        return torch.cat(output, 1)

class InceptionC1(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.branch3x3a = BasicConv2d(input_channel, 192, kernel_size=1)
        self.branch3x3b = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7a = BasicConv2d(input_channel, 192, kernel_size=1)
        self.branch7x7b = BasicConv2d(192, 192, kernel_size=(1,7), padding=(0,3))
        self.branch7x7c = BasicConv2d(192, 192, kernel_size=(7,1), padding=(3,0))
        self.branch7x7d = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3a(x)
        branch3x3 = self.branch3x3b(branch3x3)

        branch7x7 = self.branch7x7a(x)
        branch7x7 = self.branch7x7b(branch7x7)
        branch7x7 = self.branch7x7c(branch7x7)
        branch7x7 = self.branch7x7d(branch7x7)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        output = [branch3x3, branch7x7, branch_pool]
        return torch.cat(output, 1)

class InceptionC2(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channel, 320, kernel_size=1)

        self.branch3x3_1a = BasicConv2d(input_channel, 384, kernel_size=1)
        self.branch3x3_1ba = BasicConv2d(384, 384, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_1bb = BasicConv2d(384, 384, kernel_size=(3,1), padding=(1,0))

        self.branch3x3_2a = BasicConv2d(input_channel, 448, kernel_size=1)
        self.branch3x3_2b = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3_2ca = BasicConv2d(384, 384, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_2cb = BasicConv2d(384, 384, kernel_size=(3,1), padding=(1,0))

        self.branch_pool = BasicConv2d(input_channel, 192, kernel_size=1)
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3_1 = self.branch3x3_1a(x)
        branch3x3_1 = [ self.branch3x3_1ba(branch3x3_1), self.branch3x3_1bb(branch3x3_1)]
        branch3x3_1 = torch.cat(branch3x3_1, 1)

        branch3x3_2 = self.branch3x3_2a(x)
        branch3x3_2 = self.branch3x3_2b(branch3x3_2)
        branch3x3_2 = [self.branch3x3_2ca(branch3x3_2), self.branch3x3_2cb(branch3x3_2)]
        branch3x3_2 = torch.cat(branch3x3_2, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        output = [branch1x1, branch3x3_1, branch3x3_2, branch_pool]
        return torch.cat(output, 1)



class BasicConv2d(nn.Module):
    def __init__(self, input_channel, output_channel, **kargs):
        super().__init__()
        self.conv=nn.Conv2d(input_channel, output_channel, **kargs)
        self.bn = nn.BatchNorm2d(output_channel)

    def forward(self,x):
        return F.relu(self.bn(self.conv(x)),inplace=True)
