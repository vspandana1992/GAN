""" SpandanaVemulapalli : Generative Adversarial Network """

import torch
from torch import nn

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _Generative(nn.Module):
    def __init__(self):
        super(_Generative, self).__init__()
        self.GNET = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )
    def forward(self, input):
        output = self.GNET(input)
        return output
Generative = _Generative()
Generative.apply(weights_init)

class _Discriminative(nn.Module):
    def __init__(self):
        super(_Discriminative, self).__init__()
        self.DNET = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(32, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 1, 4, 2, 0, bias = False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.DNET(input)
        return output.view(-1)
Discriminative = _Discriminative()
Discriminative.apply(weights_init)
