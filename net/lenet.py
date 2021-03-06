#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.name = 'lenet'
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)
    
    def forward(self,x):
        """ x: batchsize * channel * h * w
        x : 4 * 3 * 32 * 32
        conv1: 4 * 6 * 28 * 28
        pool: 4 * 6 * 14 * 14
        conv2: 4 * 16 * 10 * 10
        pool: 4 * 16 * 5 * 5
        fc1: 4 * 120
        fc2: 4 * 84
        fc3: 4 * 10
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    net = LeNet()
    print(net)
