#!/usr/bin/env python
# -*- coding:utf-8 -*-

from skimage import transform
import numpy as np
import torch

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        if (h,w) == (new_h,new_w):
            return sample

        image = transform.resize(image, (new_h,new_w), mode='constant')
        # fix skimage resize side-effect
        image = image * 255
        image = image.astype(np.uint8)
        return (image,label)


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        if isinstance(output_size,int):
            self.output_size = (output_size,output_size)
        else:
            self.output_size = output_size

    def __call__(self,sample):
        image, label = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        assert new_w<=w
        assert new_h<=h

        top = np.random.randint(0,h-new_h+1)
        left = np.random.randint(0,w-new_w+1)

        image = image[top:top+new_h,left:left+new_w]

        return (image, label)

class RandomHorizontalFilp(object):
    def __init__(self, p=0.5):
        assert p>=0 and p<=1
        self.p = p
    
    def __call__(self, sample):
        image, label = sample
        
        assert isinstance(image, np.ndarray) and image.ndim in (2,3)
        if(np.random.random()<=self.p):
            #image = np.flip(image,1).copy()
            image = image[:,::-1,:].copy()
        return (image, label)


class ToTensor(object):
    """Convert a sample {'image':``numpy.ndarray``, 'label':int}  to {'image':tensor,'label':tensor}
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    
    """
    def __call__(self,sample):
        image, label=sample
        assert isinstance(image, np.ndarray)  and (image.ndim == 3)

        # numpy: h * w * c
        # torch.Tensor: c * h * w
        image = torch.from_numpy(image.transpose((2,0,1)))
        if isinstance(image, torch.ByteTensor):
            image = image.float().div(255)
        return (image, label)

class Norm(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        """
        Args:
            sample(Tensor,label)
        """
        image, label = sample

        assert torch.is_tensor(image) and image.ndimension()==3

        for t,m,s in zip(image,self.mean,self.std):
            t.sub_(m).div_(s)
        return (image,label)
        

if __name__ == '__main__':
    pass
