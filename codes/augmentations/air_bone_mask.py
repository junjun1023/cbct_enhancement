"""
reference: https://www.kaggle.com/pestipeti/custom-albumentation-dicomwindowshift"""

import numpy as np
import random
import albumentations as albu
from . import base

def air_bone_mask(img, **kwargs):
        """
        img is bounded [0, 1]
        """
        x_min = -500
        x_max = 500
        
        air_window = (-500, -499)
        upper = ((air_window[1]) - (x_min))/(x_max-(x_min))
        lower = ((air_window[0]) - (x_min))/(x_max-(x_min))
        air = base.hu_clip(img, (lower, upper), None, True)
        
        bone_window = (255, 256)
        upper = ((bone_window[1]) - (x_min))/(x_max-(x_min))
        lower = ((bone_window[0]) - (x_min))/(x_max-(x_min))
        bone = base.hu_clip(img, (lower, upper), None, True)
    
        
        return np.stack((air, bone))
    
    
    
def get_air_bone_mask():
        """Construct preprocessing transform
        
        Args:
                preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
                transform: albumentations.Compose
        
        """
        
        _transform = [
            albu.Lambda(image=air_bone_mask),
        ]
        return albu.Compose(_transform)
