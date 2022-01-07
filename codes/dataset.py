import os
import glob
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset as BaseDataset

from .utils import read_dicom, hu_window, valid_slices, hu_clip


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
    def __init__(self, path, intensity_aug=None, geometry_aug=None, preprocessing=None):
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
            
        # set both cbct and ct to same WL and WW (0, 1000)
#         self.xs = [hu_window(cbct, window_level=WL, window_width=WW,  show_hist=False) for cbct in self.xs]
#         self.ys = [hu_window(ct, window_level=0, window_width=1000,  show_hist=False) for ct in self.ys]
        
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug
        self.preprocessing = preprocessing
        
    
    def __getitem__(self, i):

        # read img
        x = self.xs[i].pixel_array.copy()
        y = self.ys[i].pixel_array.copy()
        

        crop_size = (64, 448)
        x = x[crop_size[0]:crop_size[1], crop_size[0]:crop_size[1]]
        y = y[crop_size[0]:crop_size[1], crop_size[0]:crop_size[1]]
        

        if self.intensity_aug:
            sample = self.intensity_aug(image=x, image0=y)
            x = np.squeeze(sample["image"])
            y = np.squeeze(sample["image0"])
            
            
        if self.geometry_aug:
            sample = self.geometry_aug(image=x, mask=y)
            x, y = sample["image"], sample["mask"]
            
        
        x = np.expand_dims(x, 0).astype(np.float32)
        y = np.expand_dims(y, 0).astype(np.float32)
        
        """
        crop_size = (64, 448)
        x = x[:, crop_size[0]:crop_size[1], crop_size[0]:crop_size[1]]
        y = y[:, crop_size[0]:crop_size[1], crop_size[0]:crop_size[1]]
        """
        
        x_f = self.xs[i].pixel_array.copy()
        y_f = self.ys[i].pixel_array.copy()

        _xs = []
        _ys = []
        hu_range = [(-512, -257), (-256, -1), (0, 255), (256, 511)]
        for hu in hu_range:
            _x = hu_clip(x_f, hu[1], hu[0], True)
            _y = hu_clip(y_f, hu[1], hu[0], True)
            _x = np.expand_dims(_x, 0)
            _y = np.expand_dims(_y, 0)
            
            _xs += [_x]
            _ys += [_y]

        x1, x2, x3, x4 = _xs
        y1, y2, y3, y4 = _ys
            
        return x, y, x1, y1, x2, y2, x3, y3, x4, y4
    
        
    def __len__(self):
        return len(self.xs)
    