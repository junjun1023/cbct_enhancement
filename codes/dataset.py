import os
import glob
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset as BaseDataset

from .utils import read_dicom, valid_slices, hu_clip, get_mask, bounded, min_max_normalize


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
        
        # read cbct and ct
        for i in range(0, len(paths), 2):
            cbct_slices = read_dicom(paths[i])
            ct_slices = read_dicom(paths[i+1])

            region = valid_slices(cbct_slices, ct_slices)
            # ditch first and last 3
            self.xs = self.xs + cbct_slices[region[0] + 3: region[1] - 3]
            self.ys = self.ys + ct_slices[region[0] + 3: region[1] - 3]
            
        
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug
        self.preprocessing = preprocessing

        
    
    def __getitem__(self, i):

        # read img
        x = self.xs[i].pixel_array.copy()
        y = self.ys[i].pixel_array.copy()
        
        # split to multiple bins
        _xs = []
        _ys = []
        hu_range = [(-512, -257), (-256, -1), (0, 255), (256, 511)]
        for hu in hu_range:
            _x = hu_clip(x, hu[1], hu[0], True)
            _y = hu_clip(y, hu[1], hu[0], True)
            _x = np.expand_dims(_x, 0)
            _y = np.expand_dims(_y, 0)
            
            _xs += [_x]
            _ys += [_y]

        x1, x2, x3, x4 = _xs
        y1, y2, y3, y4 = _ys
        
        mask_x = get_mask(x1.squeeze())
        mask_y = get_mask(y1.squeeze())
        
#         x1 = x1 * mask_x
#         y1 = y1 * mask_y
#         x2 = x2 * mask_x
#         y2 = y2 * mask_y
#         x3 = x3 * mask_x
#         y3 = y3 * mask_y
#         x4 = x4 * mask_x
#         y4 = y4 * mask_y       
        
        if self.intensity_aug:
            sample = self.intensity_aug(image=x, image0=y)
            x = np.squeeze(sample["image"])
            y = np.squeeze(sample["image0"])

        
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        if not bounded(x, (0, 1)):
            x = min_max_normalize(x)
        if not bounded(y, (0, 1)):
            y = min_max_normalize(y)
            
        x = x * mask_x
        y = y * mask_y

        # air mask
        assert x_min == -500, "make a new hu value for cbct"
        assert x_max == 500, "make a new hu value for cbct"
        upper = ((-256) - (x_min))/(x_max-(x_min))
        lower = ((-257) - (x_min))/(x_max-(x_min))
        air_x = hu_clip(x, upper, lower, True)
        
        assert y_min == -500, "make a new hu value for ct"
        assert y_max == 500, "make a new hu value for ct"
        upper = ((-256) - (y_min))/(y_max-(y_min))
        lower = ((-257) - (y_min))/(y_max-(y_min))
        air_y = hu_clip(y, upper, lower, True)
        
        # bone mask
        upper = ((256) - (x_min))/(x_max-(x_min))
        lower = ((255) - (x_min))/(x_max-(x_min))
        bone_x = hu_clip(x, upper, lower, True)

        upper = ((256) - (y_min))/(y_max-(y_min))
        lower = ((255) - (y_min))/(y_max-(y_min))
        bone_y = hu_clip(y, upper, lower, True)
            
            
            
        if self.geometry_aug:
            sample = self.geometry_aug(image=x, image0=y, image1=air_x, image2=air_y, image3=bone_x, image4=bone_y)
            x, y, air_x, air_y, bone_x, bone_y = sample["image"], sample["image0"], sample["image1"], sample["image2"], sample["image3"], sample["image4"]
            
        x = np.expand_dims(x, 0).astype(np.float32)
        y = np.expand_dims(y, 0).astype(np.float32)
        air_x = np.expand_dims(air_x, 0).astype(np.float32)
        air_y = np.expand_dims(air_y, 0).astype(np.float32)
        bone_x = np.expand_dims(bone_x, 0).astype(np.float32)
        bone_y = np.expand_dims(bone_y, 0).astype(np.float32)      

        return x, y, air_x, air_y, bone_x, bone_y
    
        
    def __len__(self):
        return len(self.xs)
    