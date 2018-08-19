#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.name = 'alexnet'
        self.conv1 = nn.Conv2d(3,96,kernel_size=(11,11),stride=4,padding=2)
        self.pool = nn.MaxPool2d((3,3),stride=2)
        self.conv2 = nn.Conv2d(96,256,kernel_size=(5,5),stride=1,padding=2)
        self.conv3 = nn.Conv2d(256,384,kernel_size=(3,3),stride=1,padding=1)
        self.conv4 = nn.Conv2d(384,384,kernel_size=(3,3),stride=1,padding=1)
        self.conv5 = nn.Conv2d(384,256,kernel_size=(3,3),stride=1,padding=1)
        self.fc1 = nn.Linear(6*6*256,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,num_classes)
        

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    net = AlexNet()
    print(net)
