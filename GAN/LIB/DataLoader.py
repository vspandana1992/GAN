""" SpandanaVemulapalli : Generative Adversarial Network """

import torchvision,torch
from torchvision import transforms

class DataLoader(object):
    def __call__(self,dataset,root):
        if dataset == 'cifar10':
            train = torchvision.datasets.CIFAR10(root, train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
            test  = torchvision.datasets.CIFAR10(root, train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
            train = [tr for tr,lbl in train if lbl==5]
            test  = [te for te,lbl in test if lbl==5]
            all   = torch.utils.data.ConcatDataset([train,test])
            return all

DL = DataLoader()
