#!/usr/bin/env python
#-*- coding:utf-8 -*-

import torch
from dataloader import Cifar10DataSet, T
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.utils.data import DataLoader

def imshow(img):
    img = img/2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img,(1,2,0)))
    plt.show()

def main():
    use_cuda = torch.cuda.is_available()
    classify_net = torch.load('model/model-vgg16-epoch4.pkl')
    classify_net.eval()
    batch_size = 4
    transform = torchvision.transforms.Compose(
        [T.Rescale(256), T.RandomCrop(224), T.RandomHorizontalFilp(),T.ToTensor(),T.Norm((0.5,0.5,0.5),(0.5,0.5,0.5))])
    cifar10_test_dataset = Cifar10DataSet('/home/tony/codes/data/cifar10/', 'test_label.txt',transform=transform)
    testloader = DataLoader(cifar10_test_dataset,batch_size = batch_size,
            shuffle=False,num_workers=2)
    classnum = cifar10_test_dataset.classnum
    correct_classes = [0 for i in range(classnum)]
    total_classes = [0 for i in range(classnum)]

    with torch.no_grad():
        for data in testloader:
            inputs,labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(),labels.cuda()
            outputs = classify_net(inputs)
            _,pred = torch.max(outputs,1)
            #print('GT', ' '.join('{:5s}'.format(cifar10_test_dataset.classes[labels[j]]) for j in range(batch_size)))
            #print('PRED', ' '.join('{:5s}'.format(cifar10_test_dataset.classes[pred[j]]) for j in range(batch_size)))
            #correct+=np.sum(pred.numpy()==labels.numpy())
            c = (pred==labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                correct_classes[label] += c[i].item()
                total_classes[label]+=1

            print('tested on {}'.format(np.sum(total_classes)),end='\r')
    print()
    print('test finished, total acc: {:.2f}%'.format(np.sum(correct_classes)*100/np.sum(total_classes)))
    for i in range(classnum):
        print('acc of {}: {:.2f}%'.format(cifar10_test_dataset.classes[i],100 * correct_classes[i]/total_classes[i]))

if __name__ == '__main__':
    main()
