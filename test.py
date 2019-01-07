#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from dataloader import Cifar10DataSet, T, CatvsDogDataSet
import net
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',"--root_dir",type=str,default='/home/tony/codes/data/catvsdog/',help="data dir")
    parser.add_argument('-l',"--list_file",type=str,default='test_list.txt',help="train label file")
    parser.add_argument('-r',"--result_file",type=str,default='result/test_res.csv',help="file path to record result")
    parser.add_argument("--check_point",type=str,default='model/model-resnet152-epoch12.pkl',help="model check point file to use")
    parser.add_argument("--use_cuda",action="store_true",default=True,help="choose to use cuda or not")
    parser.add_argument("--batch_size",type=int,default=32,help="batch size during training")
    args = parser.parse_args()
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    return args

def imshow(img):
    img = img/2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img,(1,2,0)))
    plt.show()

def main():
    args = parse_args()
    
    transform = torchvision.transforms.Compose(
        [T.Rescale((224,224)),T.ToTensor(),T.Norm((0.5,0.5,0.5),(0.5,0.5,0.5))])
    dataset = CatvsDogDataSet(args.root_dir,args.list_file,transform=transform)
    testloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False)

    classify_net = net.resnet152(pretrained=False,num_classes=dataset.classnum)
    classify_net.load_state_dict(torch.load(args.check_point))
    classify_net.eval()
    if args.use_cuda:
        classify_net.cuda()

    res_out = open(args.result_file,"w")
    res_out.write("id,label\n")
    cnt = 0
    if dataset.has_label:
        correct_classes = [0 for i in range(dataset.classnum)]
        total_classes = [0 for i in range(dataset.classnum)]
        with torch.no_grad():
            for data in testloader:
                inputs,labels = data
                if args.use_cuda:
                    inputs, labels = inputs.cuda(),labels.cuda()
                outputs = classify_net(inputs)
                _,pred = torch.max(outputs,1)
                for j in range(args.batch_size):
                    cnt+=1
                    res_out.write(str(cnt)+","+dataset.classes[pred[j]]+"\n")
                c = (pred==labels).squeeze()
                for i in range(args.batch_size):
                    label = labels[i]
                    correct_classes[label] += c[i].item()
                    total_classes[label]+=1

                print('tested on {}'.format(np.sum(total_classes)),end='\r')
        print()
        print('test finished, total acc: {:.2f}%'.format(np.sum(correct_classes)*100/np.sum(total_classes)))
        for i in range(dataset.classnum):
            print('acc of {}: {:.2f}%'.format(dataset.classes[i],100 * correct_classes[i]/total_classes[i]))
    else:
        with torch.no_grad():
            for data in testloader:
                inputs = data
                if args.use_cuda:
                    inputs = inputs.cuda()
                outputs = classify_net(inputs)
                _,pred = torch.max(outputs,1)
                for j in range(len(pred)):
                    cnt+=1
                    res_out.write(str(cnt)+","+dataset.classes[pred[j]]+"\n")
                print('tested on {}'.format(cnt),end='\r')
        print()
        print('test finished')

if __name__ == '__main__':
    main()
