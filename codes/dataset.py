import os
import glob
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset as BaseDataset

from .utils import hu_clip, read_dicom, valid_slices
from .augmentations.base import refine_mask
from .augmentations.mask import get_mask
from .augmentations.air_bone_mask import get_air_bone_mask


class Dataset(BaseDataset):
    """
    Args:
        path (str): path to dataset
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(self, path, geometry_aug=None):
        paths = sorted(glob.glob(path))
        self.xs = []
        self.ys = []
        
        # read cbct and ct
        for i in range(0, len(paths), 2):
            cbct_slices = read_dicom(paths[i])
            ct_slices = read_dicom(paths[i+1])

            region = valid_slices(cbct_slices, ct_slices)
            # ditch first and last 3
            self.xs = self.xs + cbct_slices[region[0] + 3: region[1] - 3]
            self.ys = self.ys + ct_slices[region[0] + 3: region[1] - 3]
            

        self.geometry_aug = geometry_aug

        
    
    def __getitem__(self, i):

        # read img
        x = self.xs[i].pixel_array.copy()
        y = self.ys[i].pixel_array.copy()

        mask_x = get_mask()(image=x)["image"]
        mask_y = get_mask()(image=y)["image"]
 
        x = hu_clip(x, 500, -500, True)
        y = hu_clip(y, 500, -500, True)

        x = x * mask_x
        y = y * mask_y

        sample = get_air_bone_mask()(image=x)["image"]
        air_x, bone_x = sample[0, :, :], sample[1, :, :]
        sample = get_air_bone_mask()(image=y)["image"]
        air_y, bone_y = sample[0, :, :], sample[1, :, :]
        
        bone = refine_mask(bone_x, bone_y)
        
            
        if self.geometry_aug:
            sample = self.geometry_aug(image=x, image0=y, image1=air_x, image3=bone)
            x, y, air_x, bone = sample["image"], sample["image0"], sample["image1"], sample["image2"]
            
            
        x = np.expand_dims(x, 0).astype(np.float32)
        y = np.expand_dims(y, 0).astype(np.float32)
        air_x = np.expand_dims(air_x, 0).astype(np.float32)
        bone = np.expand_dims(bone, 0).astype(np.float32)

        return x, y, air_x, bone
    
        
    def __len__(self):
        return len(self.xs)