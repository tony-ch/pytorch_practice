#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__=['Inception_v2']


class BasicConv2d(nn.Module):
    def __init__(self, input_channel, output_channel, **kargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,output_channel, **kargs)
        self.bn = nn.BatchNorm2d(output_channel)

    def forward(self,x):
        return F.relu(self.bn(self.conv(x)),inplace=True)


class InceptionBlock(nn.Module):
    def __init__(self, input_channel, fliter_nums):
        super().__init__()

        assert isinstance(fliter_nums, (list,tuple))
        assert len(fliter_nums)==6

        fnum1x1,fnum_redu3x3,fnum3x3,fnum_redu5x5,fnum5x5,pool_proj=fliter_nums

        self.branch1x1 = BasicConv2d(input_channel,fnum1x1,kernel_size=1)

        self.branch3x3a = BasicConv2d(input_channel,fnum_redu3x3, kernel_size=1)
        self.branch3x3b = BasicConv2d(fnum_redu3x3, fnum3x3, kernel_size=3,padding=1)

        self.branch5x5a = BasicConv2d(input_channel, fnum_redu5x5, kernel_size=1)
        self.branch5x5b = BasicConv2d(fnum_redu5x5, fnum5x5, kernel_size=3 ,padding=1)
        self.branch5x5c = BasicConv2d(fnum5x5, fnum5x5, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(input_channel, pool_proj, kernel_size =1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3a(x)
        branch3x3 = self.branch3x3b(branch3x3)

        branch5x5 = self.branch5x5a(x)
        branch5x5 = self.branch5x5b(branch5x5)
        branch5x5 = self.branch5x5c(branch5x5)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding =1)
        branch_pool = self.branch_pool(branch_pool)

        features = [ branch1x1, branch3x3,branch5x5,branch_pool]
        return torch.cat(features, 1)


class InceptionDownSamBlock(nn.Module):
    def __init__(self, input_channel, fliter_nums):
        super().__init__()

        assert isinstance(fliter_nums, (list,tuple))
        assert len(fliter_nums)==4

        fnum_redu3x3,fnum3x3,fnum_redu5x5,fnum5x5=fliter_nums

        self.branch3x3a = BasicConv2d(input_channel,fnum_redu3x3, kernel_size=1)
        self.branch3x3b = BasicConv2d(fnum_redu3x3, fnum3x3, kernel_size=3, stride=2, padding=1)

        self.branch5x5a = BasicConv2d(input_channel, fnum_redu5x5, kernel_size=1)
        self.branch5x5b = BasicConv2d(fnum_redu5x5, fnum5x5, kernel_size=3, padding=1)
        self.branch5x5c = BasicConv2d(fnum5x5, fnum5x5, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch3x3 = self.branch3x3a(x)
        branch3x3 = self.branch3x3b(branch3x3)

        branch5x5 = self.branch5x5a(x)
        branch5x5 = self.branch5x5b(branch5x5)
        branch5x5 = self.branch5x5c(branch5x5)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding =1)

        features = [branch3x3,branch5x5,branch_pool]
        return torch.cat(features,1)

class InceptionAux(nn.Module):
    def __init__(self, input_channel, class_nums):
        super().__init__()
        # 14x14
        self.conv = BasicConv2d(input_channel,128,kernel_size=1)
        # 4x4
        self.fc1 = nn.Linear(128*4*4,1024)
        self.fc2 = nn.Linear(1024,class_nums)

    def forward(self, x):
        # 14x14
        x = F.avg_pool2d(x,kernel_size=5,stride=3)
        # 4x4
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.7, training = self.training)
        x = self.fc2(x)
        return x

class Inception_v2(nn.Module):
    def __init__(self,input_channel=3,output_channel=10):
        super().__init__()
        self.name = "inception_v2"
        self.conv1_7x7 = BasicConv2d(input_channel,64,kernel_size=7,stride=2,padding=3)

        self.conv2_redu_3x3 = BasicConv2d(64,64,kernel_size=1)
        self.conv2_3x3 = BasicConv2d(64,192,kernel_size=3,padding=1)

        self.inception_3a = InceptionBlock(192,[64,64,64,64,96,32])
        self.inception_3b = InceptionBlock(256,[64,64,96,64,96,64])

        self.inception_4a = InceptionDownSamBlock(320,[128,160,64,96])
        self.inception_4b = InceptionBlock(576,[224,64,96,96,128,128])
        self.inception_4c = InceptionBlock(576,[192,96,128,96,128,128])
        self.inception_4d = InceptionBlock(576,[160,128,160,128,160,96])
        self.inception_4e = InceptionBlock(576,[96,128,192,160,192,96])

        self.inception_5a = InceptionDownSamBlock(576,[128,192,256,256])
        self.inception_5b = InceptionBlock(1024,[352,192,320,160,224,128])
        self.inception_5c = InceptionBlock(1024,[352,192,320,160,224,128])

        self.linear = nn.Linear(1024,output_channel)

    def forward(self,x):
        # 224x224x3
        conv1 = self.conv1_7x7(x)
        # 112x112x64
        conv1 = F.max_pool2d(conv1,kernel_size=3,stride=2,padding=1)

        # 56x56x64
        conv2 = self.conv2_redu_3x3(conv1)
        conv2 = self.conv2_3x3(conv2)
        # 56x56x192
        conv2 = F.max_pool2d(conv2,kernel_size=3,stride=2,padding=1)

        # 28x28x192
        mixed_3a = self.inception_3a(conv2)
        # 28x28x256
        mixed_3b = self.inception_3b(mixed_3a)
        # 28x28x320

        mixed_4a = self.inception_4a(mixed_3b)
        # 14x14x576
        mixed_4b = self.inception_4b(mixed_4a)
        mixed_4c = self.inception_4c(mixed_4b)
        mixed_4d = self.inception_4d(mixed_4c)
        mixed_4e = self.inception_4e(mixed_4d)

        mixed_5a = self.inception_5a(mixed_4e)
        # 7x7x1024
        mixed_5b = self.inception_5b(mixed_5a)
        mixed_5c = self.inception_5b(mixed_5b)

        features = F.avg_pool2d(mixed_5c,kernel_size=7)
        # 1x1x1024
        features = F.dropout(features, p=0.4, training=self.training, inplace=True)
        features = features.view(features.size(0),-1)

        result = self.linear(features)
        # 1x1xOUTPUT_CHANNEL
        return result


if __name__ =='__main__':
    net = Inception_v2()
    print(net)
