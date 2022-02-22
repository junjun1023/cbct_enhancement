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
from .augmentations.denoise_mask import DenoiseMask
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
    def __init__(self, path, intensity_aug=None, geometry_aug=None, identity=False, electron=False):
        paths = sorted(glob.glob(path))
        self.xs = []
        self.ys = []
        
        # read cbct and ct
        for i in range(0, len(paths), 2):
            cbct_slices = read_dicom(paths[i+1])
            ct_slices = read_dicom(paths[i])

            region = valid_slices(cbct_slices)
            # ditch first and last 3
            self.xs = self.xs + cbct_slices[region[0] + 3: region[1] - 3]
            self.ys = self.ys + ct_slices[region[0] + 3: region[1] - 3]

        self.identity = identity
        self.electron = electron
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug

        
    
    def __getitem__(self, i):

        # read img
        x = self.xs[i].pixel_array.copy()
        y = self.ys[i].pixel_array.copy()

        bound = (-512, -257)
#         if self.electron:
#             y = (y + 1015.1)/1005.8
#             x = (x + 1052.1)/1053.4
#             bound = (0.4, 0.5)

#         mask_x = DenoiseMask()(image=x, bound=bound)["image"]
#         mask_y = DenoiseMask()(image=y, bound=bound)["image"]
        mask_x = get_mask()(image=x)["image"]
        mask_y = get_mask()(image=y)["image"]
 
        bound = (-500, 500)
#         if self.electron:
#             bound = (0.5, 1.5)
        x = hu_clip(x, bound, None, True)
        y = hu_clip(y, bound, None, True)

        x = x * mask_x
        y = y * mask_y

        sample = get_air_bone_mask()(image=x)["image"]
        air_x, bone_x = sample[0, :, :], sample[1, :, :]
        sample = get_air_bone_mask()(image=y)["image"]
        air_y, bone_y = sample[0, :, :], sample[1, :, :]
        
        bone_x = refine_mask(bone_x, bone_y)

        
        if self.geometry_aug:
            sample = self.geometry_aug(image=x, image0=y, image1=air_x, image2=bone_x, image3=air_y, image4=bone_y)
            x, y, air_x, bone_x, air_y, bone_y = sample["image"], sample["image0"], \
                                                                                                                        sample["image1"], sample["image2"], \
                                                                                                                        sample["image3"], sample["image4"]
                                               
            
        if self.intensity_aug:
            sample = self.intensity_aug(image=x, image0=y, image1=air_x, image2=bone_x, image3=air_y, image4=bone_y)
            x, y, air_x, bone_x, air_y, bone_y = sample["image"], sample["image0"], \
                                                                                                                        sample["image1"], sample["image2"], \
                                                                                                                        sample["image3"], sample["image4"]
            

        x = np.expand_dims(x, 0).astype(np.float32)
        y = np.expand_dims(y, 0).astype(np.float32)
        air_x = np.expand_dims(air_x, 0).astype(np.float32)
        bone_x = np.expand_dims(bone_x, 0).astype(np.float32)
        air_y = np.expand_dims(air_y, 0).astype(np.float32)
        bone_y = np.expand_dims(bone_y, 0).astype(np.float32)
    
        if self.identity:
            return y, y, air_y, bone_y, air_y, bone_y
 

        return x, y, air_x, bone_x, air_y, bone_y
    
        
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
    def __init__(self, cbct_path, ct_path, ditch=3, intensity_aug=None, geometry_aug=None, identity=False):

        # read cbct and ct
        assert cbct_path.split("_")[0] == ct_path.split("_")[0]         
        self.patient_id = cbct_path.split("_")[0].split("/")[-1]
        
        cbct_slices = read_dicom(cbct_path)
        ct_slices = read_dicom(ct_path)

        region = valid_slices(cbct_slices)
        # ditch first and last 2
        # make ct more than cbct
        self.xs = cbct_slices[region[0] + ditch: region[1] - ditch]
        self.ys = ct_slices[region[0] + ditch: region[1] - ditch]
        
        self.identity = identity
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug

        
    def __getitem__(self, i):

        # read img
        x = self.xs[i].pixel_array.copy()
        y = self.ys[i].pixel_array.copy()

        bound = (-512, -257)
        mask_x = get_mask()(image=x)["image"]
        mask_y = get_mask()(image=y)["image"]
 
        bound = (-500, 500)
        x = hu_clip(x, bound, None, True)
        y = hu_clip(y, bound, None, True)

        x = x * mask_x
        y = y * mask_y

        sample = get_air_bone_mask()(image=x)["image"]
        air_x, bone_x = sample[0, :, :], sample[1, :, :]
        sample = get_air_bone_mask()(image=y)["image"]
        air_y, bone_y = sample[0, :, :], sample[1, :, :]
        
        bone_x = refine_mask(bone_x, bone_y)

        
        if self.geometry_aug:
            sample = self.geometry_aug(image=x, image0=y, image1=air_x, image2=bone_x, image3=air_y, image4=bone_y)
            x, y, air_x, bone_x, air_y, bone_y = sample["image"], sample["image0"], \
                                                                                                                        sample["image1"], sample["image2"], \
                                                                                                                        sample["image3"], sample["image4"]
                                               
            
        if self.intensity_aug:
            sample = self.intensity_aug(image=x, image0=y, image1=air_x, image2=bone_x, image3=air_y, image4=bone_y)
            x, y, air_x, bone_x, air_y, bone_y = sample["image"], sample["image0"], \
                                                                                                                        sample["image1"], sample["image2"], \
                                                                                                                        sample["image3"], sample["image4"]
            

        x = np.expand_dims(x, 0).astype(np.float32)
        y = np.expand_dims(y, 0).astype(np.float32)
        air_x = np.expand_dims(air_x, 0).astype(np.float32)
        bone_x = np.expand_dims(bone_x, 0).astype(np.float32)
        air_y = np.expand_dims(air_y, 0).astype(np.float32)
        bone_y = np.expand_dims(bone_y, 0).astype(np.float32)
    
        if self.identity:
            return y, y, air_y, bone_y, air_y, bone_y
 

        return x, y, air_x, bone_x, air_y, bone_y
    
        
    def __len__(self):
        return len(self.xs)
    
    
    def patientID(self):
        return self.patient_id
    
    
    

def DicomsDataset(path, geometry_aug=None, intensity_aug=None, identity=False):
        paths = sorted(glob.glob(path))
        
        datasets = []
        # read cbct and ct
        for i in range(0, len(paths), 2):
            scans = DicomDataset(cbct_path=paths[i+1], ct_path=paths[i], ditch=3, geometry_aug=geometry_aug, intensity_aug=intensity_aug, identity=identity)
            datasets = datasets + [scans]
            
        datasets = ConcatDataset(datasets)
        return datasets

    
