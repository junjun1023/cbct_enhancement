import os
import glob
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset as BaseDataset

from .utils import read_npy, valid_slices, hu_clip, get_mask, bounded, min_max_normalize


class Dataset(BaseDataset):
    """
    Args:
        path (str): path to dataset
        x_channel (int): number of channels for X(images)
        y_channel (int): number of channels for y(masks)
        do_resize (tuple of ints): resize the image and mask, if None, DO NOT do resize
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(self, path, intensity_aug=None, geometry_aug=None, preprocessing=None, tolerance=20):
        paths = sorted(glob.glob(path))
        self.xs = []
        self.ys = []
        self.air = []
        self.bone = []
        
        # read cbct and ct
        for i in range(0, len(paths), 8):
            # air, bone, img, mask
            air = read_npy(paths[i])
            bone = read_npy(paths[i+1])
            cbct_slices =  read_npy(paths[i+2])
            ct_slices = read_npy(paths[i]+6)

            region = valid_slices(cbct, ct)
            # ditch first and last 3
            self.xs = self.xs + cbct_slices[region[0] + 3: region[1] - 3]
            self.ys = self.ys + ct_slices[region[0] + 3: region[1] - 3]
            self.air = self.air + air[region[0] + 3: region[1] - 3]
            self.bone = self.bone + bone[region[0] + 3: region[1] - 3]
            
        
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug
        self.preprocessing = preprocessing

        
    
    def __getitem__(self, i):

        # read img
        x = self.xs[i]
        y = self.ys[i]
        air = self.air[i]
        bone = self.bone[i]

        if self.geometry_aug:
            sample = self.geometry_aug(image=x, image0=y, image1=air, image2=bone)
            x, y, air, bone = sample["image"], sample["image0"], sample["image1"], sample["image2"]
            
        x = np.expand_dims(x, 0).astype(np.float32)
        y = np.expand_dims(y, 0).astype(np.float32)
        air = np.expand_dims(air, 0).astype(np.float32)
        bone = np.expand_dims(bone, 0).astype(np.float32)

        return x, y, air, bone
    
        
    def __len__(self):
        return len(self.xs)
    