#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage import io
import os
from dataloader import custom_transform as T


class ClassificationDataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        super().__init__()
        self.transform = transform
        list_file = os.path.join(root_dir,list_file)
        assert os.path.exists(list_file)

        self.root_dir = root_dir
        sample_list = [x.strip('\n') for x in open(list_file).readlines()]
        self.img_list = [ x.split()[0] for x in sample_list]
        self.label_list = [ int(x.split()[1]) for x in sample_list]
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        # img: h*w*c numpy array, RGB mode
        img = io.imread(os.path.join(self.root_dir, self.img_list[idx]))
        label = self.label_list[idx]
        sample = (img, label)

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample

class Cifar10DataSet(ClassificationDataset):
    def __init__(self, root_dir, list_file, transform=None):
        super().__init__(root_dir,list_file,transform)
        self.classes=('plane','car','bird','cat','deer','dog','frog','horse',
        'ship','truck')
        self.classnum=10

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    
    cifar10_dataset = Cifar10DataSet('/home/tony/codes/data/cifar10/', 'train_label.txt')
    classes = cifar10_dataset.classes
    #for i in range(len(cifar10_dataset)):
    for i in range(1):
        sample = cifar10_dataset[i]
        img,label = sample
        print('label: {}'.format(classes[label]))
        plt.imshow(img)
        plt.show()
   
    data_loader = DataLoader(cifar10_dataset,batch_size=4, shuffle=True, num_workers=0)
    for i_, sample_ in enumerate(data_loader):
        print(i_, sample_[0].size(), sample_[1], sample_[1].size())

        if i_ == 2:
            plt.imshow(sample_[0][0])
            plt.show()
            break
    
    transform = transforms.Compose(
        [T.RandomHorizontalFilp(),T.ToTensor(),T.Norm((0.5,0.5,0.5),(0.5,0.5,0.5))])

    cifar10_dataset = Cifar10DataSet('/home/tony/codes/data/cifar10/', 'train_label.txt',transform=transform)
    
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()
    for i in range(1):
        sample = cifar10_dataset[i]
        img, label = sample
        print('label: {}'.format(label))
        imshow(img)
    
    data_loader = DataLoader(cifar10_dataset,batch_size=4, shuffle=True, num_workers=0)
    for i_, sample_ in enumerate(data_loader):
        print(i_, sample_[0].size(), sample_[1], sample_[1].size())

        if i_ == 2:
            imshow(sample_[0][0])
            break
    
    transform = transforms.Compose(
       [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,
       download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size = 4,
       shuffle=True,num_workers=0)
    for i_, sample_ in enumerate(trainloader):
        #img, label = sample
        print(i_, sample_[0].size(), sample_[1], sample_[1].size())

        if i_ == 2:
            imshow(sample_[0][0])
            break
