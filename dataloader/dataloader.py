#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,
        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size = 4,
        shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=False,transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
        shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse',
        'ship','truck')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('{:5s}'.format(classes[labels[j]]) for j in range(4)))
