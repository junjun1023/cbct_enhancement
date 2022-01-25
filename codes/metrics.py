import numpy as np

import torch
import torch.nn as nn
from .utils import find_mask
from segmentation_models_pytorch.utils.metrics import Fscore, IoU

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

    
class SNR:
    def __init__(self):
        self.name = "SNR"
        
    @staticmethod
    def __call__(pr, gt):
        '''
        input tensors of same shape

        return SNR
        '''
        upper = torch.sum(torch.pow(gt, 2))
        lower = torch.sum(torch.pow(gt - pr, 2))
        
        return 10 * torch.log10(upper / lower)
    
    
    
class ContourEval:
    def __init__(self):
        self.name = "ContourEval"
    
    @staticmethod
    def __call__(pr, gt, cnt_width=1, mode="dice"):
        
        if isinstance(pr, torch.Tensor):
            pr = pr.squeeze().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.squeeze().cpu().numpy()
        
        pr_cnt = find_mask(pr, cnt_width, False)
        gt_cnt = find_mask(gt, cnt_width, False)
        
        score = None
        if mode == "dice":
            score = Fscore()(torch.from_numpy(pr_cnt).float(), torch.from_numpy(gt_cnt).float())
        elif mode == "iou":
            score = IoU()(torch.from_numpy(pr_cnt).float(), torch.from_numpy(gt_cnt).float())
        else:
            assert 0, "Unknown evaluation. Need to eval with Fscore or IoU"
        
        return score
            
            
        
    