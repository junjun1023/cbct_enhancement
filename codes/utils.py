import glob
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, 'gray')
    plt.show()
    
    
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


def find_mask(img, width=1, plot=False):
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    assert len(img.shape) == 2, "Supportive for 2-dims only"
     
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
    cv2.drawContours(img, [c], 0, (255, 255, 255), width)
    img = img[:, :, 0].astype(np.float32) / 255
    
    return img
    

    
def grow_mask_outward(img, kernel=(5, 5), iterations = 1):
    # https://stackoverflow.com/questions/55948254/scale-contours-up-grow-outward
    kernel = np.ones(kernel, np.uint8)
    img = cv2.dilate(img, kernel, iterations = iterations)
    img = find_mask(img, -1, False)
    return img
    
    
    
def get_mask(img):
    if np.max(img) == 0:
        return np.zeros(img.shape, dtype=np.float32)
    mask = find_mask(img, -1, False)
    mask = grow_mask_outward(mask)
    return mask


def read_dicom(path):
    g = glob.glob(os.path.join(path, '*.dcm'))
    slices = [pydicom.read_file(s) for s in g]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices


def hu_clip_tensor(img, sourceBound, targetBound=None, min_max_norm=True):
    lower = sourceBound[0]
    upper = sourceBound[1]
    img = torch.where(img < lower, lower, img)
    img = torch.where(img > upper, upper, img)
    if min_max_norm:
        img = min_max_normalize(img, sourceBound, targetBound)
        
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


def hu_window(scan, window_level=40, window_width=80, min_max_norm=True):
    scan = scan.pixel_array.copy()
    window = [window_level-window_width//2, window_width//2-window_level]
    
    scan = np.where(scan < window[0], window[0], scan)
    scan = np.where(scan > window[1], window[1], scan)
    
    if min_max_norm:
        scan = min_max_normalize(scan,(upper, lower))

    return scan


def show_raw_pixel(slices):
    #讀出像素值並且儲存成numpy的格式
    image = hu_window(slices, window_level=0, window_width=1000,  show_hist=False)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
    
    
    
# return  a region where cbct images aren't all black
def valid_slices(cbcts):
    found_start = False
    start = 0
    end = len(cbcts)-1
    
    # iterate through cbct slices, and find which regions aren't all black (-1000)
    for idx, sli in enumerate(cbcts):
        image = sli.pixel_array
        
        if not found_start and len(np.unique(image))!=1:
            start = idx
            found_start = True
                                   
        elif found_start and len(np.unique(image)) == 1:
            end = idx
            break
        
    return start, end
