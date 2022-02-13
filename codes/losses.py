import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim
from .extractor import Extractor
from .activation import Activation


class SSIMLoss:

    def __init__(self):
        self.__name__ = "SSIMLoss"

    @staticmethod
    def __call__(img1, img2, data_range=1.0, size_average=True):
        _ssim = 1 - ssim(img1, img2, data_range=data_range, size_average=size_average) # return (N,)
        return _ssim

    
class MAELoss(nn.L1Loss):
    def __init__(self):
        nn.L1Loss.__init__(self)
        self.__name__ = 'MAELoss'
        
        
class PerceptualLoss:

    def __init__(self, feature_list=["features_23"], normalize=True):
        self.__name__ = "PerceptualLoss"
        self.extractor = Extractor(None, feature_list)
        self.normalize = normalize
        self.feature_list = feature_list
    
    def __call__(self, x, y):
        
        assert x.size()[1] == 1 or x.size()[1] == 3, "Can't match channel number"
        
        if x.size()[1] == 1:
            x = torch.cat((x, x, x), dim=1)
        if y.size()[1] == 1:
            y = torch.cat((y, y, y), dim=1)
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        
        if self.normalize:
            for ch in range(3):
                x[:, ch, :, :] = (x[:, ch, :, :] - mean[ch])/std[ch]
                y[:, ch, :, :] = (y[:, ch, :, :] - mean[ch])/std[ch]
            
        x = self.extractor(x)["features_23"]
        y = self.extractor(y)["features_23"]

        return nn.L1Loss()(x, y)
        
        
class MultiScaleHeads(nn.Module):
    
    class Flatten(nn.Module):
        def forward(self, x):
            return torch.flatten(x, start_dim=1)
    
    def __init__(self, n_classes=1, channels=(64, 128, 256, 512, 1024), activation=None):
        
        super(MultiScaleHeads, self).__init__()
        self.channels = channels
        
        modules = []
        for channel in channels:
            seq = [
                nn.AdaptiveAvgPool2d(1),
                self.Flatten(),
                nn.Linear(in_features=channel, out_features=n_classes),
            ]
            module = nn.Sequential(*seq)
            modules += [module]
            
        self.classifiers = nn.ModuleList(modules)
        self.act = Activation(name=activation)
        
    def forward(self, x):
        
        ret = []
        for feat, classifier in zip(x, self.classifiers):
            f = classifier(feat)
            f = self.act(f)
            ret += [f]
            
        return ret