#!/usr/bin/env python
#-*- coding:utf-8 -*-

import torch.optim as optim
from net import LeNet,AlexNet,VGG16,Inception_v1,Inception_v2,Inception_v1_bn,Inception_v3,Xception
from dataloader import Cifar10DataSet, T
# import dataloader.custom_transform as T
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    use_cuda = torch.cuda.is_available()
    print(">>> building net")
    classify_net = Inception_v2()
    classify_net.train()
    print(classify_net)

    if use_cuda:
        print(">>> using cuda now")
        classify_net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classify_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.RMSprop(classify_net.parameters(), lr=0.05, alpha=0.99, eps=1.0, weight_decay=0.9, momentum=0.9)
    max_epoch = 4
    output_step = 20

    transform = transforms.Compose(
        [T.Rescale(320), T.RandomCrop(299),T.RandomHorizontalFilp(),T.ToTensor(),T.Norm((0.5,0.5,0.5),(0.5,0.5,0.5))])
    cifar10_train_dataset = Cifar10DataSet('/home/tony/codes/data/cifar10/', 'train_label.txt',transform=transform)
    trainloader = DataLoader(cifar10_train_dataset,batch_size = 24,
       shuffle=True,num_workers=0)

    print(">>> start training")
    for epoch in range(max_epoch):
        runing_loss = 0.0
        #for i, data in enumerate(dataloader.trainloader,0):
        for i, data in enumerate(trainloader,0):
            inputs,labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(),labels.cuda()

            optimizer.zero_grad()
            output = classify_net(inputs)
            loss = criterion(output,labels)
            #aux_loss_1 = criterion(aux_1,labels)
            #aux_loss_2 = criterion(aux_2,labels)
            #loss = res_loss+0.3*aux_loss_1+0.3*aux_loss_2
            loss.backward()

            optimizer.step()

            runing_loss += loss.item()
            lr = optimizer.param_groups[0]['lr']
            if i%output_step == output_step-1:
                print('[epoch: {:3d}, step: {:5d}] loss: {:.3f} lr: {:.6f}'.format(epoch+1,i+1,runing_loss/output_step, lr))
                runing_loss = 0.0
    print('>>> finished training')
    torch.save(classify_net,'model/model-{}-epoch{}.pkl'.format(classify_net.name,max_epoch))
    #torch.save(classify_net.state_dict(),'model/model-epoch{}.pkl'.format(max_epoch))

if __name__ == '__main__':
    main()
