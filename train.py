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
from utils import AugTransform, get_activation, save_checkpoint, aggrAugs, train, val
from utils import init_weights_xavier, init_weights_normal, init_weights_kaiming, init_weights_LSUV
from Res import BigRes


device = torch.device("cuda")
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


train_aug = iaa.Sequential([iaa.SomeOf((0, 3), [iaa.Affine(scale=(0.5, 1.5)), 
                                                               iaa.Affine(translate_percent={'x':(-0.2, 0.2), 'y':(-0.2, 0.2)}), 
                                                               iaa.Affine(rotate=(-45, 45)),
                                                               iaa.Affine(shear=(-16, 16)),
                                                               iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                                                               iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.1, 0.1)))])], random_order=True) 
test_aug = [iaa.ScaleX((1.2)), iaa.Fliplr(1.0)]


CIFAR_train = torchvision.datasets.FashionMNIST(root=".", train=True, transform=[resize((448, 448)), ToTensor()], target_transform=None, download=True)
CIFAR_test = torchvision.datasets.FashionMNIST(root=".", train=False, transform=[resize((448, 448)), ToTensor()], target_transform=None, download=True)



Res50 = BigRes(image_channels=3)
Res50.to(device)

Res50 = init_weights_LSUV(Res50, CIFAR_train, n_points=128)




train_dataloader = torch.utils.data.DataLoader(CIFAR_train, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(CIFAR_test, batch_size=16, shuffle=True)
loss_fn = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
optimizer = AdamW(Res50.parameters(), lr=0.001, weight_decay=0)




Res50.conv1.register_forward_hook(get_activation("conv1_f"))
Res50.conv1.register_backward_hook(get_activation("conv1_b"))
Res50.layer1[0].conv3.register_forward_hook(get_activation("layer1[0]_conv3_f"))
Res50.layer1[0].conv3.register_backward_hook(get_activation("layer1[0]_conv3_b"))
Res50.layer2[1].conv2.register_forward_hook(get_activation("layer2[1]_conv2_f"))
Res50.layer2[1].conv2.register_backward_hook(get_activation("layer2[1]_conv2_b"))
Res50.layer3[3].conv3.register_forward_hook(get_activation("layer3[3]_conv3_f"))
Res50.layer3[3].conv3.register_backward_hook(get_activation("layer3[3]_conv3_b"))
Res50.layer4[2].conv3.register_forward_hook(get_activation("layer4[2]_conv3_f"))
Res50.layer4[2].conv3.register_backward_hook(get_activation("layer4[2]_conv3_b"))
writer = SummaryWriter("logs")



epochs = 20
train_losses = []
val_losses = []
best_accuracy = 0
is_best = False
state_dict = {}




for epoch in range(epochs):
    train_loss = train(Res50, train_dataloader, loss_fn, optimizer, augment = train_aug, aug_prob=0.5)
    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_histogram("layer1.Weights", Res50.layer1[1].conv1.weight)
    writer.add_histogram("layer3.Weights", Res50.layer3[3].conv3.weight)
    writer.add_histogram("layer4.Weights", Res50.layer4[2].conv2.weight)
    
    
#     writer.add_histogram("conv1.factivation", activation['conv1_f'], epoch)
#     writer.add_histogram("conv1.bactivation", activation['conv1_f'][0], epoch)
#     writer.add_histogram("layer1.factivation", activation['layer1[0]_conv3_f'], epoch)
#     writer.add_histogram("layer1.bactivation", activation['layer1[0]_conv3_b'][0], epoch)
#     writer.add_histogram("layer2.factivation", activation['layer2[1]_conv2_f'], epoch)
#     writer.add_histogram("layer2.bactivation", activation['layer2[1]_conv2_b'][0], epoch)
#     writer.add_histogram("layer3.factivation", activation['layer3[3]_conv3_f'], epoch)
#     writer.add_histogram("layer3.bactivation", activation['layer3[3]_conv3_b'][0], epoch)
#     writer.add_histogram("layer4.factivation", activation['layer4[2]_conv3_f'], epoch)
#     writer.add_histogram("layer4.bactivation", activation['layer4[2]_conv3_b'][0], epoch)
    
    val_loss, val_accuracy = val(Res50, test_dataloader, loss_fn, optimizer, test_aug)
    writer.add_scalar("loss/eval", val_loss, epoch)
    writer.add_scalar("accuracy", val_accuracy, epoch)
    writer.flush()
  
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        is_best = True
    state_dict = {
      'model_state': Res50.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'epoch': epoch
    }
    save_checkpoint(state_dict, is_best, './Checkpoints')
