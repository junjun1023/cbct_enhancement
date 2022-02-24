"""
reference: https://www.kaggle.com/pestipeti/custom-albumentation-dicomwindowshift
"""

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


def air_bone_mask(img, bound, air_bound, bone_bound, min_max_norm):
        """
        img is bounded [0, 1]
        """
        x_min = bound[0]
        x_max = bound[1]

        upper = ((air_bound[1]) - (x_min))/(x_max-(x_min))
        lower = ((air_bound[0]) - (x_min))/(x_max-(x_min))
        air = hu_clip(img, (lower, upper), None, min_max_norm)

        upper = ((bone_bound[1]) - (x_min))/(x_max-(x_min))
        lower = ((bone_bound[0]) - (x_min))/(x_max-(x_min))
        bone = hu_clip(img, (lower, upper), None, min_max_norm)

        return np.stack((air, bone))


class AirBoneMask(ImageOnlyTransform):
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
            bound=(-500, 500),
            air_bound=(-500, -499),
            bone_bound=(255, 256),
            min_max_norm=True,
            always_apply=False,
            p=0.5,
    ):
        super(AirBoneMask, self).__init__(always_apply, p)
        
        assert len(bound) ==  2, "Not enough / too many args of bound for img"        
        assert len(air_bound) ==  2, "Not enough / too many args of bound for air mask"
        assert len(bone_bound) ==  2, "Not enough / too many args of bound for bone mask"
        
        self.bound = bound
        self.air_bound = air_bound
        self.bone_bound = bone_bound
        self.min_max_norm = min_max_norm
        
    def apply(self, image, bound=(), air_bound=(), bone_bound=(), min_max_norm=True, **params):
        return air_bone_mask(image, self.bound, self.air_bound, self.bone_bound, self.min_max_norm)

    
    def get_params_dependent_on_targets(self, params):
        return {"bound": self.bound,
                "air_bound": self.air_bound, 
                "bone_bound": self.bone_bound, 
                "min_max_norm": self.min_max_norm}
    

    @property
    def targets_as_params(self):
        return ["image"]
    

    def get_transform_init_args_names(self):
        return "bound", "air_bound", "bone_bound", "min_max_norm"

    
