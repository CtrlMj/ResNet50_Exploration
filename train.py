import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from imgaug import augmenters as iaa
from cv2 import resize
import sys
import matplotlib.pyplot as plt
%matplotlib inline
sys.path.insert(1, "./LSUV-pytorch") # ---> for LSUV initialization approach from https://github.com/ducha-aiki/LSUV-pytorch
from LSUV import LSUVinit

device = torch.device("cuda")
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
