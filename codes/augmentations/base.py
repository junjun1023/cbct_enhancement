import numpy as np
import cv2
import torch
from segmentation_models_pytorch.utils.metrics import IoU

def bounded(img, bound):
    if img.min() < bound[0] or img.max() > bound[1]:
        return False
    return True


def min_max_normalize(img, sourceBound=None, targetBound=None):
    
    if sourceBound == None:
        sourceBound = (img.min(), img.max())

    if sourceBound[0] == sourceBound[1]:
        if isinstance(img, np.ndarray):
            return np.zeros(img.shape, dtype=np.float32)
        elif isinstance(img, torch.Tensor):
            return torch.where(img > 0, 0)

    img = (img - sourceBound[0])/(sourceBound[1] - sourceBound[0])
    if targetBound:
        img = img * (targetBound[1] - targetBound[0]) + targetBound[0]

    return img



def hu_clip(img, sourceBound, targetBound=None, min_max_norm=True, zipped=False):
    lower = sourceBound[0]
    upper = sourceBound[1]
    img = np.where(img < lower, lower, img)
    img = np.where(img > upper, upper, img)

    if min_max_norm:
        img = min_max_normalize(img, sourceBound, targetBound)
    
    if zipped:
        img = (img*255).astype(np.uint8).astype(np.float32) / 255
        
    return img


def find_mask(img, plot=False):
    
    assert len(img.shape) == 2, "Only gray scale image is acceptable"
     
    img = img.copy()
    if np.max(img) <= 1:
        img = (img * 255).astype(np.uint8)
    
    img[img != 0] = 255
    
    cnts, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)

    if plot:
        img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(img, [c], 0, (255, 255, 255), 3)
        plt.figure(0, figsize=(6,6))
        plt.imshow(img)
        plt.show()
        
    img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(img, [c], 0, (255, 255, 255), -1)
    img = img[:, :, 0].astype(np.float32) / 255
    
    return img



def grow_mask_outward(img, kernel=(5, 5), iterations = 1):
    # https://stackoverflow.com/questions/55948254/scale-contours-up-grow-outward
    kernel = np.ones(kernel, np.uint8)
    img = cv2.dilate(img, kernel, iterations = iterations)
    img = find_mask(img, False)
    return img

    

def get_mask(img):
    if np.max(img) == 0:
        return np.zeros(img.shape, dtype=np.float32)
    mask = find_mask(img, False)
    mask = grow_mask_outward(mask)
    return mask



def refine_mask(bone_x, bone_y):

    bone_x = bone_x.copy()
    bone_y = bone_y.copy()
    
    bone_x = (bone_x * 255).astype(np.uint8)
    bone_y = (bone_y * 255).astype(np.uint8)
    
    kernel = np.ones((9, 9), np.uint8)
    dilate_x = cv2.dilate(bone_x, kernel, iterations = 1)
    dilate_y = cv2.dilate(bone_y, kernel, iterations = 1)
    
    cnts, hier = cv2.findContours(dilate_x.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = dilate_y.shape
    
    comps = []
    for cnt in cnts:
        empty = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.drawContours(empty, [cnt], 0, (255, 255, 255), -1)
        empty = empty.astype(np.float32) / 255
        empty = empty[:, :, 0]
        empty_y = dilate_y * empty

        iou = IoU()(torch.from_numpy(empty_y), torch.from_numpy(empty))

        if iou > 0.01:
            comps += [empty]
        else:
            pass

    comps = np.stack(comps, axis=-1)
    comps = comps.max(-1)
    processed = bone_x * comps
    processed = processed.astype(np.float32) / 255

    return processed
    