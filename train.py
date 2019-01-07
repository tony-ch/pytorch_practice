#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch.optim as optim
import net
from dataloader import Cifar10DataSet, T, CatvsDogDataSet
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',"--root_dir",type=str,default='/home/tony/codes/data/catvsdog/',help="data dir")
    parser.add_argument('-l',"--list_file",type=str,default='train_all_label.txt',help="train label file")
    parser.add_argument("--pretrained",action="store_true",default=True,help="use pretrained model or not")
    parser.add_argument("--use_cuda",action="store_true",default=True,help="choose to use cuda or not")
    parser.add_argument("--lr",type=float,default=0.001,help="base learning rate")
    parser.add_argument("--batch_size",type=int,default=32,help="batch size during training")
    parser.add_argument("--max_epoch",type=int,default=16,help="max epoch to train")
    parser.add_argument("--output_step",type=int,default=20,help="frequency to print log")
    args = parser.parse_args()
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    return args


def main():
    args = parse_args()
    print(">>> create dataloader")
    transform = transforms.Compose(
        [T.Rescale((256,256)), T.RandomCrop(224),T.RandomHorizontalFilp(),T.ToTensor(),T.Norm((0.5,0.5,0.5),(0.5,0.5,0.5))])
    dataset = CatvsDogDataSet(args.root_dir, args.list_file,transform=transform)
    trainloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True)
    
    print(">>> building net")
    classify_net = net.resnet152(pretrained=args.pretrained, num_classes=dataset.classnum)
    classify_net.train()
    print(classify_net)
    if args.use_cuda:
        print(">>> using cuda now")
        classify_net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classify_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(classify_net.parameters(),lr=0.005)
    #optimizer = optim.RMSprop(classify_net.parameters(), lr=0.05, alpha=0.99, eps=1.0, weight_decay=0.9, momentum=0.9)

    print(">>> start training")
    for epoch in range(args.max_epoch):
        runing_loss = 0.0
        #for i, data in enumerate(dataloader.trainloader,0):
        for i, data in enumerate(trainloader,0):
            inputs,labels = data
            if args.use_cuda:
                inputs, labels = inputs.cuda(),labels.cuda()

            optimizer.zero_grad()
            output = classify_net(inputs)
            loss = criterion(output,labels)
            loss.backward()

            optimizer.step()

            runing_loss += loss.item()
            lr = optimizer.param_groups[0]['lr']
            if i%args.output_step == args.output_step-1:
                print('[epoch: {:3d}, step: {:5d}] loss: {:.5f} lr: {:.6f}'.format(epoch+1,i+1,runing_loss/args.output_step, lr))
                runing_loss = 0.0
        if epoch%4 == 3:
            torch.save(classify_net.state_dict(),'model/model-{}-epoch{}.pkl'.format(classify_net.name,epoch+1))
    print('>>> finished training')

if __name__ == '__main__':
    main()
