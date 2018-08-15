#!/usr/bin/env python
#-*- coding:utf-8 -*-

import torch.optim as optim
from net import net
from data import dataloader
import torch.nn as nn
import torch


def main():
    use_cuda = torch.cuda.is_available()
    classify_net = net.Net()
    if use_cuda:
        classify_net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classify_net.parameters(), lr=0.001, momentum=0.9)
    max_epoch = 3

    for epoch in range(max_epoch):
        runing_loss = 0.0
        for i, data in enumerate(dataloader.trainloader,0):
            if use_cuda:
                data.cuda()
            inputs,labels = data

            optimizer.zero_grad()
            output = classify_net(inputs)
            loss = criterion(output,labels)
            loss.backward()

            optimizer.step()

            runing_loss += loss.item()
            
            if i%2000 == 1999:
                print('[epoch: {:3d}, step: {:5d}] loss: {:.3f}'.format(epoch+1,i+1,runing_loss/2000.0))
                runing_loss = 0.0
    print('finished training')
    torch.save(classify_net,'model/model-epoch{}.pkl'.format(max_epoch))
    #torch.save(classify_net.state_dict(),'model/model-epoch{}.pkl'.format(max_epoch))

if __name__ == '__main__':
    main()
