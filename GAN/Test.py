""" SpandanaVemulapalli : Generative Adversarial Network : Test Model"""

import torch
from LIB.GAN import Generative,Discriminative
from PIL import Image as ImPIL
import numpy as np
from torchvision import transforms

Ws = torch.load("./model.t7")
Model = Generative
input = torch.randn(1, 100, 1, 1)
Model.load_state_dict(Ws)
Im = transforms.ToPILImage()
Im(Model(input).squeeze(0))
