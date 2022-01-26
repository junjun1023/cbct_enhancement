"""
reference: https://www.kaggle.com/pestipeti/custom-albumentation-dicomwindowshift"""

import numpy as np
import random
import albumentations as albu
from . import base


def _mask(img, **kwargs):
    """
    need origin dicom image
    """
    window = (-512, -257)
    img = base.hu_clip(img, (window[0], window[1]), None, True)
    img = base.get_mask(img)
    return img

    

def get_mask():
        """Construct preprocessing transform
        
        Args:
                preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
                transform: albumentations.Compose
        
        """
        
        _transform = [
            albu.Lambda(image=_mask),
        ]
        return albu.Compose(_transform)
