import os
import glob
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import ConcatDataset

from .utils import hu_clip, read_dicom, valid_slices, min_max_normalize
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
    def __init__(self, path, intensity_aug=None, geometry_aug=None, mode="both"):
        paths = sorted(glob.glob(path))
        self.xs = []
        self.ys = []
        
        # read cbct and ct
        for i in range(0, len(paths), 2):
            cbct_slices = read_dicom(paths[i])
            ct_slices = read_dicom(paths[i+1])

            region = valid_slices(cbct_slices)
            # ditch first and last 3
            self.xs = self.xs + cbct_slices[region[0] + 3: region[1] - 3]
            self.ys = self.ys + ct_slices[region[0] + 3: region[1] - 3]
            
        
        self.mode = mode
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug

        
    
    def __getitem__(self, i):

        # read img
        x = self.xs[i].pixel_array.copy()
        y = self.ys[i].pixel_array.copy()

        mask_x = get_mask()(image=x)["image"]
        mask_y = get_mask()(image=y)["image"]
 
        x = hu_clip(x, (-500, 500), None, True)
        y = hu_clip(y, (-500, 500), None, True)

        x = x * mask_x
        y = y * mask_y

        sample = get_air_bone_mask()(image=x)["image"]
        air_x, bone_x, tissue_x = sample[0, :, :], sample[1, :, :], sample[2, :, :]
        sample = get_air_bone_mask()(image=y)["image"]
        air_y, bone_y, tissue_y = sample[0, :, :], sample[1, :, :], sample[2, :, :]
        
        bone_x = refine_mask(bone_x, bone_y)

        
        if self.geometry_aug:
            sample = self.geometry_aug(image=x, image0=y, image1=air_x, image2=bone_x, image3=air_y, image4=bone_y, image5=tissue_x, image6=tissue_y)
            x, y, air_x, bone_x, air_y, bone_y, tissue_x, tissue_y = sample["image"], sample["image0"], \
                                                                                                                        sample["image1"], sample["image2"], \
                                                                                                                        sample["image3"], sample["image4"], \
                                                                                                                        sample["image5"], sample["image6"]
            
        if self.intensity_aug:
            sample = self.intensity_aug(image=x, image0=y, image1=air_x, image2=bone_x, image3=air_y, image4=bone_y, image5=tissue_x, image6=tissue_y)
            x, y, air_x, bone_x, air_y, bone_y, tissue_x, tissue_y = sample["image"], sample["image0"], \
                                                                                                                        sample["image1"], sample["image2"], \
                                                                                                                        sample["image3"], sample["image4"], \
                                                                                                                        sample["image5"], sample["image6"]
            

        x = np.expand_dims(x, 0).astype(np.float32)
        y = np.expand_dims(y, 0).astype(np.float32)
        air_x = np.expand_dims(air_x, 0).astype(np.float32)
        bone_x = np.expand_dims(bone_x, 0).astype(np.float32)
        air_y = np.expand_dims(air_y, 0).astype(np.float32)
        bone_y = np.expand_dims(bone_y, 0).astype(np.float32)
        tissue_x = np.expand_dims(tissue_x, 0).astype(np.float32)
        tissue_y = np.expand_dims(tissue_y, 0).astype(np.float32)   
        
        if self.mode == "cbct":
            return x, air_x, bone_x, tissue_x
        elif self.mode == "ct":
            return y, air_y, bone_y, tissue_y
        elif self.mode == "unpaired":
            return x, y, air_x, bone_x, tissue_x
        elif self.mode == "both":
            return x, y, air_x, bone_x, tissue_x, air_y, bone_y, tissue_y

        return x, y, air_x, bone_x, tissue_x, air_y, bone_y, tissue_y
    
        
    def __len__(self):
        return len(self.xs)
    
    
    
    
class DicomDataset(BaseDataset):
    """
    Args:
        path (str): path to dataset
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(self, cbct_path, ct_path, ditch=2, intensity_aug=None, geometry_aug=None):

        # read cbct and ct
        cbct_slices = read_dicom(cbct_path)
        ct_slices = read_dicom(ct_path)

        region = valid_slices(cbct_slices)
        # ditch first and last 2
        # make ct more than cbct
        self.xs = cbct_slices[region[0] + ditch: region[1] - ditch]
        self.ys = ct_slices[region[0]: region[1]]
        self.ditch = ditch
        
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug

        
    def __getitem__(self, i):

        # read img
        x = self.xs[i].pixel_array.copy()
        y = self.ys[i+2].pixel_array.copy()
        y_p1 = self.ys[i+1].pixel_array.copy()
        y_p2 = self.ys[i].pixel_array.copy()
        y_n1 = self.ys[i+3].pixel_array.copy()
        y_n2 = self.ys[i+4].pixel_array.copy()

        mask_x = get_mask()(image=x)["image"]
        mask_y = get_mask()(image=y)["image"]
        mask_y_p1 = get_mask()(image=y_p1)["image"]
        mask_y_p2 = get_mask()(image=y_p1)["image"]
        mask_y_n1 = get_mask()(image=y_n1)["image"]
        mask_y_n2 = get_mask()(image=y_n2)["image"]
 
        x = hu_clip(x, 500, -500, True)
        y = hu_clip(y, 500, -500, True)
        y_p1 = hu_clip(y_p1, 500, -500, True)
        y_p2 = hu_clip(y_p2, 500, -500, True)
        y_n1 = hu_clip(y_n1, 500, -500, True)
        y_n2 = hu_clip(y_n2, 500, -500, True)
        
        x = x * mask_x
        y = y * mask_y
        y_p1 = y_p1 * mask_y_p1
        y_p2 = y_p2 * mask_y_p2
        y_n1 = y_n1 * mask_y_n1
        y_n2 = y_n2 * mask_y_n2
        
        del mask_x, mask_y, mask_y_p1, mask_y_p2, mask_y_n1, mask_y_n2

        sample = get_air_bone_mask()(image=x)["image"]
        air_x, bone_x = sample[0, :, :], sample[1, :, :]
        sample = get_air_bone_mask()(image=y)["image"]
        _, bone_y = sample[0, :, :], sample[1, :, :]
        
        bone_x = refine_mask(bone_x, bone_y)
        
            
        if self.geometry_aug:
            sample = self.geometry_aug(image=x, image0=y, image1=air_x, image2=bone_x, image3=y_p2, image4=y_p1, image5=y_n1, image6=y_n2)
            x, y, air_x, bone_x, y_p2, y_p1, y_n1, y_n2 = sample["image"], sample["image0"], sample["image1"], sample["image2"], \
                                                                                                        sample["image3"], sample["image4"], sample["image5"], sample["image6"]
        
        if self.intensity_aug:
            sample = self.intensity_aug(image=x, image0=air_x, image1=bone)
            x, air_x, bone = sample["image"], sample["image0"], sample["image1"]
            
        x = np.expand_dims(x, 0).astype(np.float32)
        y = np.expand_dims(y, 0).astype(np.float32)
        air_x = np.expand_dims(air_x, 0).astype(np.float32)
        bone_x = np.expand_dims(bone_x, 0).astype(np.float32)
        y_p1 = np.expand_dims(y_p1, 0).astype(np.float32)
        y_p2 = np.expand_dims(y_p2, 0).astype(np.float32)
        y_n1 = np.expand_dims(y_n1, 0).astype(np.float32)
        y_n2 = np.expand_dims(y_n2, 0).astype(np.float32)
        

        return x, y, air_x, bone_x, y_p2, y_p1, y_n1, y_n2
    
        
    def __len__(self):
        return len(self.xs)
    
    
    
def DicomsDataset(path, geometry_aug=None):
        paths = sorted(glob.glob(path))
        
        datasets = []
        # read cbct and ct
        for i in range(0, len(paths), 2):
            scans = DicomDataset(cbct_path=paths[i], ct_path=paths[i+1], ditch=2, geometry_aug=geometry_aug)
            datasets = datasets + [scans]
            
        datasets = ConcatDataset(datasets)
        return datasets
    
    
    
class UnpairedDataset(BaseDataset):
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
        
        # read cbct and ct
        for i in range(0, len(paths)):
            slices = read_dicom(paths[i])

            region = valid_slices(cbct_slices)
            # ditch first and last 3
            self.xs = self.xs + slices[region[0] + 3: region[1] - 3]
            

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
            sample = self.geometry_aug(image=x, image0=y, image1=air_x, image2=bone)
            x, y, air_x, bone = sample["image"], sample["image0"], sample["image1"], sample["image2"]
            
            
        x = np.expand_dims(x, 0).astype(np.float32)
        y = np.expand_dims(y, 0).astype(np.float32)
        air_x = np.expand_dims(air_x, 0).astype(np.float32)
        bone = np.expand_dims(bone, 0).astype(np.float32)

        return x, y, air_x, bone
    
        
    def __len__(self):
        return len(self.xs)