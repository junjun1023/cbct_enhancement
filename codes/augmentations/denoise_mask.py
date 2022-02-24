"""
reference: https://www.kaggle.com/pestipeti/custom-albumentation-dicomwindowshift"""

import numpy as np
import random
from albumentations.core.transforms_interface import ImageOnlyTransform
from .base import get_mask, min_max_normalize



def hu_clip(img, sourceBound, targetBound=None, min_max_norm=True):
    lower = sourceBound[0]
    upper = sourceBound[1]
    img = np.where(img < lower, lower, img)
    img = np.where(img > upper, upper, img)

    if min_max_norm:
        img = min_max_normalize(img, sourceBound, targetBound)
        
    return img


def denoise_mask(img, bound, min_max_normazlie):
    """
    need origin dicom image
    """
    img = hu_clip(img, (bound[0], bound[1]), None, True)
    img = get_mask(img)
    return img


class DenoiseMask(ImageOnlyTransform):
    """As preprocessing for dicom.
    Note: It won't work for preprocessed png or jpg images. Please use the dicom's HU values
    (rescaled width slope/intercept!)
    
    Args:
        bound (int, int): (lower bound of HU, upper bound of HU)
        min_max_normalize: (bool) Apply min-max normalization
    Targets:
        image
    """
    def __init__(
            self,
            bound=(-512, -257),
            min_max_norm=True,
            always_apply=False,
            p=0.5,
    ):
        super(DenoiseMask, self).__init__(always_apply, p)
        self.bound = bound
        self.min_max_norm = min_max_norm

        assert len(self.bound) ==  2, "Not enough args of bound for denoise mask"
        
    def apply(self, image, bound=(), min_max_norm=True, **params):
        return denoise_mask(image, bound, min_max_norm)

    
    def get_params_dependent_on_targets(self, params):
        return {"bound": self.bound, "min_max_norm": self.min_max_norm}
    

    @property
    def targets_as_params(self):
        return ["image"]
    

    def get_transform_init_args_names(self):
        return "bound", "min_max_norm"

    