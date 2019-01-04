#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Inception_v1', 'Inception_v1_bn']


class BasicConv2d(nn.Module):
    _use_bn=False

    def __init__(self, input_channel, output_channel, **kargs):
        super().__init__()
        if self._use_bn:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channel,output_channel, **kargs),
                nn.BatchNorm2d(output_channel)
            )
        else:
            self.conv = nn.Conv2d(input_channel, output_channel, **kargs)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

    @classmethod
    def update_use_bn(cls, use_bn):
        cls._use_bn=use_bn


class InceptionBlock(nn.Module):
    def __init__(self, input_channel, fliter_nums):
        super().__init__()

        assert isinstance(fliter_nums, (list,tuple))
        assert len(fliter_nums) == 6

        fnum1x1,fnum_redu3x3,fnum3x3,fnum_redu5x5,fnum5x5,pool_proj=fliter_nums

        self.branch1x1 = BasicConv2d(input_channel,fnum1x1,kernel_size=1)

        self.branch3x3a = BasicConv2d(input_channel,fnum_redu3x3, kernel_size=1)
        self.branch3x3b = BasicConv2d(fnum_redu3x3, fnum3x3, kernel_size=3,padding=1)

        self.branch5x5a = BasicConv2d(input_channel, fnum_redu5x5, kernel_size=1)
        self.branch5x5b = BasicConv2d(fnum_redu5x5, fnum5x5, kernel_size=5, padding=2)

        self.branch_pool = BasicConv2d(input_channel, pool_proj, kernel_size =1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3a(x)
        branch3x3 = self.branch3x3b(branch3x3)

        branch5x5 = self.branch5x5a(x)
        branch5x5 = self.branch5x5b(branch5x5)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding =1)
        branch_pool = self.branch_pool(branch_pool)

        features = [ branch1x1, branch3x3,branch5x5,branch_pool]
        return torch.cat(features, 1)

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


class Inception(nn.Module):
    def __init__(self,input_channel,output_channel, aux_logits,use_bn):
        super().__init__()
        BasicConv2d.update_use_bn(use_bn)

        self.conv1_7x7 = BasicConv2d(input_channel,64,kernel_size=7,stride=2,padding=3)

        self.conv2_redu_3x3 = BasicConv2d(64,64,kernel_size=1)
        self.conv2_3x3 = BasicConv2d(64,192,kernel_size=3,padding=1)

        self.inception_3a = InceptionBlock(192,[64,96,128,16,32,32])
        self.inception_3b = InceptionBlock(256,[128,128,192,32,96,64])

        self.inception_4a = InceptionBlock(480,[192,96,208,16,48,64])
        self.inception_4b = InceptionBlock(512,[160,112,224,24,64,64])
        self.inception_4c = InceptionBlock(512,[128,128,256,24,64,64])
        self.inception_4d = InceptionBlock(512,[112,144,288,32,64,64])
        self.inception_4e = InceptionBlock(528,[256,160,320,32,128,128])

        self.aux_logits = aux_logits
        if aux_logits:
            self.inception_aux_1 = InceptionAux(512,output_channel)
            self.inception_aux_2 = InceptionAux(528,output_channel)

        self.inception_5a = InceptionBlock(832,[256,160,320,32,128,128])
        self.inception_5b = InceptionBlock(832,[384,192,384,48,128,128])

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
        # 28x28x480
        mixed_3 = F.max_pool2d(mixed_3b,kernel_size=3,stride=2,padding=1)

        # 14x14x480
        mixed_4a = self.inception_4a(mixed_3)
        if self.aux_logits and self.training:
            aux_1 = self.inception_aux_1(mixed_4a)
        # 14x14x512
        mixed_4b = self.inception_4b(mixed_4a)
        mixed_4c = self.inception_4c(mixed_4b)
        mixed_4d = self.inception_4d(mixed_4c)
        if self.aux_logits and self.training:
            aux_2 = self.inception_aux_2(mixed_4d)
        # 14x14x528
        mixed_4e = self.inception_4e(mixed_4d)
        # 14x14x832
        mixed_4 = F.max_pool2d(mixed_4e,kernel_size=3,stride=2,padding=1)

        # 7x7x832
        mixed_5a = self.inception_5a(mixed_4)
        mixed_5b = self.inception_5b(mixed_5a)

        # 7x7x1024
        features = F.avg_pool2d(mixed_5b,kernel_size=7)
        # 1x1x1024
        features = F.dropout(features, p=0.4, training=self.training, inplace=True)
        features = features.view(features.size(0),-1)

        result = self.linear(features)
        # 1x1xOUTPUT_CHANNEL
        if self.aux_logits and self.training:
            return result, aux_1, aux_2
        else:
            return result


class Inception_v1(Inception):
    def __init__(self,input_channel=3,num_classes=10,aux_logits=True):
        super().__init__(input_channel,num_classes,aux_logits,use_bn=False)
        self.name = 'inception_v1'


class Inception_v1_bn(Inception):
    def __init__(self, input_channel=3,num_classes=10,aux_logits=True):
        super().__init__(input_channel,num_classes,aux_logits,use_bn=True)
        self.name = 'inception_v1_bn'


if __name__ =='__main__':
    net = Inception_v1()
    print(net)
