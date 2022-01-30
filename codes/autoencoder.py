# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from .activation import Activation



class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=1024, out_channels=64):
        super(Decoder, self).__init__()
        self.blocks = nn.Sequential(
            UpBlock(in_channels, 512),
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, out_channels))
              
    def forward(self, x):
        x = self.blocks(x)
        return x 
 


class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(Encoder, self).__init__()
        self.blocks = nn.Sequential(
            DownBlock(in_channels, 64),
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512),
            DownBlock(512, 1024))

    def forward(self, x):
        return self.blocks(x)


class AutoEncoder(nn.Module):  # 必要 inherit nn.Module
    def __init__(self, in_channels=1, head_channels=1):
        super().__init__()
        
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(1024, 64)
        self.head = nn.Conv2d(64, head_channels, 3, padding=1)

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)

        return x
