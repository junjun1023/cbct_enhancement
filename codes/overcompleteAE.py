# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from .activation import Activation


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.up(x)
        x = self.act(x)
        return x

    
class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.act(x)
        return x




class Decoder(nn.Module):
    def __init__(self, channels=(128, 64, 32), out_channel=1):
        super(Decoder, self).__init__()
        
        blocks = []
        for ix in range(len(channels)-1):
            blocks += [ DownBlock(channels[ix], channels[ix+1]) ]
            blocks += [nn.InstanceNorm2d(channels[ix])]
            
        blocks += [DownBlock(channels[-1], out_channel)]
        self.blocks = nn.Sequential(*blocks)
              
    def forward(self, x):
        return self.blocks(x)



class Encoder(nn.Module):
    def __init__(self, in_channel=1, channels=(32, 64, 128)):
        super(Encoder, self).__init__()
        
        blocks = [ UpBlock(in_channel, channels[0]) ]
        for ix in range(len(channels)-1):
            blocks += [nn.InstanceNorm2d(channels[ix])]
            blocks += [UpBlock(channels[ix], channels[ix+1]) ]
            
            
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


    
class OvercompleteAE(nn.Module):  # 必要 inherit nn.Module
    def __init__(self, in_channel=1, out_channel=1):
        super().__init__()
        
        self.encoder = Encoder(in_channel=in_channel, channels=(16, 32, 64))
        self.decoder = Decoder(out_channel=out_channel, channels=(64, 32, 16))

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)

        return x

