import numpy as np

import torch
import torch.nn as nn

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(pr, gt, max_val):
        
        _max = torch.pow(torch.tensor(max_val, device=pr.device), 2)
        mse = torch.mean(torch.pow(gt - pr, 2))
        
        if mse > 0:
            return 10 * torch.log10(_max / mse)
        
        return 99
