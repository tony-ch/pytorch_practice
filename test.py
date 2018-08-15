#!/usr/bin/env python
#-*- coding:utf-8 -*-

import torch
from data import dataloader
import matplotlib.pyplot as plt
import torchvision
from net import net
import numpy as np

def imshow(img):
    img = img/2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img,(1,2,0)))
    plt.show()

def main():

    # classify_net = net.Net()
    classify_net = torch.load('model/model-epoch3.pkl')
    #correct=0
    #total = 0
    correct_classes=[ 0 for i in range(10)]
    total_classes = [ 0 for i in range(10)]
    with torch.no_grad():
        for data in dataloader.testloader:
            inputs,labels = data
            outputs = classify_net(inputs)
            _,pred = torch.max(outputs,1)
            #print('GT', ' '.join('{:5s}'.format(dataloader.classes[labels[j]]) for j in range(4)))
            #print('PRED', ' '.join('{:5s}'.format(dataloader.classes[pred[j]]) for j in range(4)))
            #correct+=np.sum(pred.numpy()==labels.numpy())
            c = (pred==labels).squeeze()
            for i in range(4):
                label = labels[i]
                correct_classes[label] += c[i].item()
                total_classes[label]+=1

            #correct+=(pred==labels).sum().item()
            #total += 4
            #print('tested on {}'.format(total),end='\r')
            print('tested on {}'.format(np.sum(total_classes)),end='\r')
    print()
    print('test finished')
    #print('correct:{}, total:{}. acc:{:.2f}'.format(correct,total, correct/total))
    for i in range(10):
        print('acc of {}: {:.2f}%'.format(dataloader.classes[i],100 * correct_classes[i]/total_classes[i]))

if __name__ == '__main__':
    main()
