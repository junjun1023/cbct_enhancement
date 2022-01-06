import numpy as np
import torch
import torch.nn as nn

from pytorch_msssim import ssim

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
        
     
