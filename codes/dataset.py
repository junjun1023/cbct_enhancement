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
from .augmentations.window_mask import AirBoneMask

from .augmentations.mask import get_mask
from .augmentations.air_bone_mask import get_air_bone_mask



class DicomDataset(BaseDataset):
    """
    Args:
        path (str): path to dataset
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(self, cbct_path, ct_path, ditch=3, 
                 intensity_aug=None, geometry_aug=None, 
                 identity=False, electron=False, position="pelvic", g_coord=False, l_coord=False):

        # read cbct and ct
        assert cbct_path.split("/")[-1].split("_")[0] == ct_path.split("/")[-1].split("_")[0]     
        self.patient_id = cbct_path.split("/")[-1].split("_")[0] 
        
        cbct_slices = read_dicom(cbct_path)
        ct_slices = read_dicom(ct_path)

        region = valid_slices(cbct_slices)
        # ditch first and last 3
        self.xs = cbct_slices[region[0] + ditch: region[1] - ditch]
        self.ys = ct_slices[region[0] + ditch: region[1] - ditch]

        encoding = []
        length = (region[1] - ditch) - (region[0] + ditch)
        quotient, remainder = length // 5, length % 5

        for i in range(5):
            cnt = quotient
            if remainder > 0:
                cnt = cnt + 1
                remainder = remainder - 1
            encoding = encoding + [i for _ in range(cnt)]
        self.encoding = encoding

        self.position = position
        self.identity = identity
        self.electron = electron
        self.g_coord = g_coord
        self.l_coord = l_coord
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug

        self.x_norm = (0, 1)
        self.y_norm = (0, 1)
        if electron:
            self.y_norm = (-1015.1, 1005.8)
            if position == "pelvic" or position == "abdomen":
                self.x_norm = (-1052.1, 1053.4)
            elif position == "chest":
                self.x_norm = (-1059.6, 1067.2)
            elif position == "headneck":
                self.x_norm = (-1068.8, 1073.5)
            else:
                assert False, "Position: pelvic, abdomen, chest, and headneck"

                
        self.p_encoding = 0
        if position == "headneck":
            self.p_encoding = 0
        elif position == "chest":
            self.p_encoding = 1
        elif position == "abdomen":
            self.p_encoding = 2
        elif position == "pelvic":
            self.p_encoding = 3
        else:
            assert False, "Position: pelvic, abdomen, chest, and headneck"      
                
                
    def __getitem__(self, i):

        # read img
        x = self.xs[i].pixel_array.copy()
        y = self.ys[i].pixel_array.copy()
        encoding = self.encoding[i]
        p_encoding = self.p_encoding
        
       ############################
        # Data denoising
        ###########################  
        denoise_bound = (-512, -257)
        if self.electron:
            y = (y - self.y_norm[0])/self.y_norm[1]
            x = (x - self.x_norm[0])/self.x_norm[1]
            denoise_bound = (0.4, 0.5)
            
        mask_x = DenoiseMask(bound=denoise_bound, always_apply=True)(image=x)["image"]
        mask_y = DenoiseMask(bound=denoise_bound, always_apply=True)(image=y)["image"]
        
       ############################
        # Dicom Window Shift Augmentation
        ###########################  
        view_bound = (-500, 500)
        if self.electron:
            view_bound = (0.5, 1.5)
        x = hu_clip(x, view_bound, None, True)
        y = hu_clip(y, view_bound, None, True)

        x = x * mask_x
        y = y * mask_y

        space = view_bound[1] - view_bound[0]
       ############################
        # Get air bone mask
        ###########################          
        air_bound = (-500, -499)
        bone_bound = (255, 256)
        if self.electron:
            air_bound = (0.5, 0.5009)
            bone_bound = (1.2, 1.2009)

        sample = AirBoneMask(bound=view_bound, air_bound=air_bound, bone_bound=bone_bound, always_apply=True)(image=x)["image"]
        air_x, bone_x = sample[0, :, :], sample[1, :, :]
        sample = AirBoneMask(bound=view_bound, air_bound=air_bound, bone_bound=bone_bound, always_apply=True)(image=y)["image"]
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
    
        encoding = np.ones(x.shape, dtype=np.float32) * encoding
        p_encoding = np.ones(x.shape, dtype=np.float32) * p_encoding
        c, w, h = x.shape
        local_encoding = self.make_grid2d(w, h)
        
        if self.identity:
            coord_y = y
            if self.g_coord and self.l_coord:
                coord_y = np.concatenate((y, encoding, p_encoding, local_encoding), axis=0)
            elif self.g_coord and not self.l_coord:
                coord_y = np.concatenate((y, local_encoding), axis=0)
            elif not self.g_coord and self.l_coord:
                coord_y = np.concatenate((y, encoding, p_encoding), axis=0)
            return coord_y, y, air_y, bone_y, air_y, bone_y
 
        coord_x = x
        if self.g_coord and self.l_coord:
            coord_x = np.concatenate((x, encoding, p_encoding, local_encoding), axis=0)
        elif self.g_coord and not self.l_coord:
            coord_x = np.concatenate((x, local_encoding), axis=0)
        elif not self.g_coord and self.l_coord:
            coord_x = np.concatenate((x, encoding, p_encoding), axis=0)        

        return coord_x, y, air_x, bone_x, air_y, bone_y
    
        
    def __len__(self):
        return len(self.xs)
    
    
    def patientID(self):
        return self.patient_id
    

    def make_grid2d(self, height, width):
        h, w = height, width
        grid_x, grid_y = np.meshgrid(np.arange(0, h), np.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = np.stack((grid_x, grid_y), 0)

        return grid
    
    
    
def DicomsDataset(path, geometry_aug=None, intensity_aug=None, identity=False, electron=False, position="pelvic", g_coord=False, l_coord=False):
        paths = sorted(glob.glob(path))
        
        datasets = []
        # read cbct and ct
        for i in range(0, len(paths), 2):
            scans = DicomDataset(cbct_path=paths[i+1], ct_path=paths[i], ditch=3, 
                                 geometry_aug=geometry_aug, intensity_aug=intensity_aug, 
                                 identity=identity, electron=electron, position=position, g_coord=g_coord, l_coord=l_coord)
            datasets = datasets + [scans]
            
        datasets = ConcatDataset(datasets)
        return datasets

    

    
class DicomSegmentDataset(BaseDataset):
    """
    Args:
        path (str): path to dataset
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(self, cbct_path, ct_path, ditch=3, segment=8, 
                 intensity_aug=None, geometry_aug=None, 
                 identity=False, electron=False, position="pelvic", g_coord=False, l_coord=False):

        # read cbct and ct
        assert cbct_path.split("/")[-1].split("_")[0] == ct_path.split("/")[-1].split("_")[0]     
        self.patient_id = cbct_path.split("/")[-1].split("_")[0] 
        
        cbct_slices = read_dicom(cbct_path)
        ct_slices = read_dicom(ct_path)

        region = valid_slices(cbct_slices)
        # ditch first and last 3
        self.xs = cbct_slices[region[0] + ditch: region[1] - ditch]
        self.ys = ct_slices[region[0] + ditch: region[1] - ditch]
        
        encoding = []
        length = (region[1] - ditch) - (region[0] + ditch)
        quotient, remainder = length // 5, length % 5

        for i in range(5):
            cnt = quotient
            if remainder > 0:
                cnt = cnt + 1
                remainder = remainder - 1
            encoding = encoding + [i for _ in range(cnt)]
        self.encoding = encoding
        
        self.position = position
        self.segment = segment
        self.identity = identity
        self.electron = electron
        self.g_coord = g_coord
        self.l_coord = l_coord
        self.intensity_aug = intensity_aug
        self.geometry_aug = geometry_aug

        self.x_norm = (0, 1)
        self.y_norm = (0, 1)
        if electron:
            self.y_norm = (-1015.1, 1005.8)
            if position == "pelvic" or position == "abdomen":
                self.x_norm = (-1052.1, 1053.4)
            elif position == "chest":
                self.x_norm = (-1059.6, 1067.2)
            elif position == "headneck":
                self.x_norm = (-1068.8, 1073.5)
            else:
                assert False, "Position: pelvic, abdomen, chest, and headneck"

                
        self.p_encoding = 0
        if position == "headneck":
            self.p_encoding = 0
        elif position == "chest":
            self.p_encoding = 1
        elif position == "abdomen":
            self.p_encoding = 2
        elif position == "pelvic":
            self.p_encoding = 3
        else:
            assert False, "Position: pelvic, abdomen, chest, and headneck"      
                
                
    def __getitem__(self, idx):

        # read img
        xs = []
        air_xs = []
        bone_xs = []
        ys = []
        air_ys = []
        bone_ys = []
        encodings = []
        
        index = np.array(range(self.segment)) - self.segment // 2

        for s in index:
            i = s + idx
            
            if i < 0 or i >= self.__len__():
                x = np.zeros((512, 512), dtype=np.float32)
                y = np.zeros((512, 512), dtype=np.float32)
                encoding = 0 if i<0 else 4
                air_x = np.zeros((512, 512), dtype=np.float32)
                bone_x = np.zeros((512, 512), dtype=np.float32)
                air_y = np.zeros((512, 512), dtype=np.float32)
                bone_y = np.zeros((512, 512), dtype=np.float32)       
                
            else:
                x = self.xs[i].pixel_array.copy()
                y = self.ys[i].pixel_array.copy()
                encoding = self.encoding[i]
               ############################
                # Data denoising
                ###########################  
                denoise_bound = (-512, -257)
                if self.electron:
                    y = (y - self.y_norm[0])/self.y_norm[1]
                    x = (x - self.x_norm[0])/self.x_norm[1]
                    denoise_bound = (0.4, 0.5)

                mask_x = DenoiseMask(bound=denoise_bound, always_apply=True)(image=x)["image"]
                mask_y = DenoiseMask(bound=denoise_bound, always_apply=True)(image=y)["image"]

                view_bound = (-500, 500)
                if self.electron:
                    view_bound = (0.5, 1.5)
                x = hu_clip(x, view_bound, None, True)
                y = hu_clip(y, view_bound, None, True)

                x = x * mask_x
                y = y * mask_y

               ############################
                # Get air bone mask
                ###########################          
                air_bound = (-500, -499) # -500, -426
                bone_bound = (255, 256) # 400, 500
                if self.electron:
                    air_bound = (0.5, 0.5009)
                    bone_bound = (1.2, 1.2009)

                sample = AirBoneMask(bound=view_bound, air_bound=air_bound, bone_bound=bone_bound, always_apply=True)(image=x)["image"]
                air_x, bone_x = sample[0, :, :], sample[1, :, :]
                sample = AirBoneMask(bound=view_bound, air_bound=air_bound, bone_bound=bone_bound, always_apply=True)(image=y)["image"]
                air_y, bone_y = sample[0, :, :], sample[1, :, :]

                bone_x = refine_mask(bone_x, bone_y)

            xs += [x]
            air_xs += [air_x]
            bone_xs += [bone_x]
            ys += [y]
            air_ys += [air_y]
            bone_ys += [bone_y]
            encodings += [encoding]
        
        xs = np.stack(xs, axis=-1)
        air_xs = np.stack(air_xs, axis=-1)
        bone_xs = np.stack(bone_xs, axis=-1)
        ys = np.stack(ys, axis=-1)
        air_ys = np.stack(air_ys, axis=-1)
        bone_ys = np.stack(bone_ys, axis=-1)

        
        if self.geometry_aug:
            sample = self.geometry_aug(image=xs, image0=ys, image1=air_xs, image2=bone_xs, image3=air_ys, image4=bone_ys)
            xs, ys, air_xs, bone_xs, air_ys, bone_ys = sample["image"], sample["image0"], \
                                                                                                                        sample["image1"], sample["image2"], \
                                                                                                                        sample["image3"], sample["image4"]
                    
            
        if self.intensity_aug:
            sample = self.intensity_aug(image=xs, image0=ys, image1=air_xs, image2=bone_xs, image3=air_ys, image4=bone_ys)
            xs, ys, air_xs, bone_xs, air_ys, bone_ys = sample["image"], sample["image0"], \
                                                                                                                        sample["image1"], sample["image2"], \
                                                                                                                        sample["image3"], sample["image4"]
        
        encodings = np.ones(xs.shape, dtype=np.float32) * encodings
        encodings = np.expand_dims(np.moveaxis(encodings, -1, 0), 1)
        xs = np.expand_dims(np.moveaxis(xs, -1, 0), 1)
        ys = np.expand_dims(np.moveaxis(ys, -1, 0), 1)
        air_xs = np.expand_dims(np.moveaxis(air_xs, -1, 0), 1)
        bone_xs = np.expand_dims(np.moveaxis(bone_xs, -1, 0), 1)
        air_ys = np.expand_dims(np.moveaxis(air_ys, -1, 0), 1)
        bone_ys = np.expand_dims(np.moveaxis(bone_ys, -1, 0), 1)
        
        if self.identity:
            if self.g_coord:
                return ys, ys, air_ys, bone_ys, air_ys, bone_ys, encodings
            return ys, ys, air_ys, bone_ys, air_ys, bone_ys
        
        if self.g_coord:
            return xs, ys, air_xs, bone_xs, air_ys, bone_ys, encodings
        return xs, ys, air_xs, bone_xs, air_ys, bone_ys
    
        
    def __len__(self):
        return len(self.xs)
    
    
    def patientID(self):
        return self.patient_id
    

    def make_grid2d(self, height, width):
        h, w = height, width
        grid_x, grid_y = np.meshgrid(np.arange(0, h), np.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = np.stack((grid_x, grid_y), 0)

        return grid
    
    
    

    
def DicomsSegmentDataset(path, geometry_aug=None, intensity_aug=None, 
                         identity=False, electron=False, position="pelvic", segment=8, g_coord=False, l_coord=False):
        paths = sorted(glob.glob(path))
        
        datasets = []
        # read cbct and ct
        for i in range(0, len(paths), 2):
            scans = DicomSegmentDataset(cbct_path=paths[i+1], ct_path=paths[i], ditch=3, segment=segment, 
                                 geometry_aug=geometry_aug, intensity_aug=intensity_aug, 
                                 identity=identity, electron=electron, position=position, g_coord=g_coord, l_coord=l_coord)
            datasets = datasets + [scans]
            
        datasets = ConcatDataset(datasets)
        return datasets