#!/usr/bin/env python
#-*- coding:utf-8 -*-

import torch.optim as optim
from net import Net
from dataloader import Cifar10DataSet, T
# import dataloader.custom_transform as T
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    use_cuda = torch.cuda.is_available()
    classify_net = Net()
    if use_cuda:
        classify_net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classify_net.parameters(), lr=0.001, momentum=0.9)
    max_epoch = 4

    transform = transforms.Compose(
        [T.RandomHorizontalFilp(),T.ToTensor(),T.Norm((0.5,0.5,0.5),(0.5,0.5,0.5))])
    cifar10_train_dataset = Cifar10DataSet('/home/tony/codes/data/cifar10/', 'train_label.txt',transform=transform)
    trainloader = DataLoader(cifar10_train_dataset,batch_size = 4,
       shuffle=True,num_workers=2)

    for epoch in range(max_epoch):
        runing_loss = 0.0
        #for i, data in enumerate(dataloader.trainloader,0):
        for i, data in enumerate(trainloader,0):
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
