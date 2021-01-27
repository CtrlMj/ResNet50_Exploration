import torch
import torch.nn as nn

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_stride=1):
        super(Resblock, self).__init__()
        self.out_size = 4*out_channels
        self.conv1 = nn.Conv2d(self.in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=mid_sride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(4*out_channels)
        self.projection = None
        self.non_linearity = nn.ReLU()
        if in_channels != 4*out_channels  or mid_stride != 1:
            self.projection = nn.Sequential(nn.Conv2d(in_channels, 4*out_channels, kernel_size=1, stride=mid_stride),
                                            nn.BatchNorm2d(4*out_channels))
        
    def forward(self, input):
        x = input
        x = self.non_linearity(self.bn1(self.conv1(x)))
        x = self.non_linearity(self.bn2(self.conv2(x)))
        x = self.non_linearity(self.bn3(self.conv3(x)))
        if self.projection:
            input = self.projection(input)
        x = input + x
        return x
        



class BigRes(nn.Module):
    def __init__(self, image_channels, out_classes=10, layers=[(3, 64, 2), (4, 128, 2) (6, 256, 2), (3, 512, 2)]):
        super(BigRes, self).__init__()
        self.in_channels = layers[0][1]
        self.out_channels = 4*layers[-1][1]
        self.non_linearity = nn.ReLU()
        self.initial_layer = nn.Sequential(nn.Conv2d(image_channels, self.in_channels, kernel_size=7, padding=3, stride=2),
                                           nn.BatchNorm2d(64))
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        
        self.Res_layers = []
        for n_blocks, out_channel, stride in layers:
            layer = self.create_blocks(n_blocks, self.in_channels, out_channel, stride=1)
            layers.append(layer)
        
        self.Res_layers = nn.Sequential(self.Res_layers)
        nn.lastpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc     = nn.Linear(self.out_channels, out_classes)
        
    def forward(self, x):
        x = self.maxpool(self.non_linearity(self.initial_layer(x)))
        x = self.Res_layers(x)
        x = self.laspool(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x
    
    def create_blocks(self, n_blocks, out_channel, stride):
        blocks = []
        for i in range(n_blocks):
            block = Resblock(self.in_channels, out_channel, mid_stride=stride)
            blocks.append(block)
            self.in_channels = 4*out_channels
        return nn.Sequential(blocks)
