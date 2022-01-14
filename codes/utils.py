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


def min_max_normalize(img):
    img = (img - img.min())/(img.max()-img.min())
    return img


def find_mask(img, plot=False):
    
    assert len(img.shape) == 2, "Only gray scale image is acceptable"
 
    if np.max(img) == 1:
        img = (img * 255).astype(np.uint8)
    
    img[img != 0] = 255
    
    cnts, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)

    if plot:
        img = np.zeros((img.shape[0], img.shape[1], 3))
        cv2.drawContours(img, [c], 0, (255, 255, 255), 3)
        plt.figure(0, figsize=(6,6))
        plt.imshow(img)
        plt.show()
        
    img = np.zeros((img.shape[0], img.shape[1], 3))
    cv2.drawContours(img, [c], 0, (255, 255, 255), -1)
    img = img[:, :, 0].astype(np.float32) / 255
    
    return img
    

def grow_mask_outward(img, kernel=(5, 5)):
    # https://stackoverflow.com/questions/55948254/scale-contours-up-grow-outward
    kernel = np.ones(kernel, np.uint8)
    img = cv2.dilate(img, kernel, iterations = 1)
    img = find_mask(img, False)
    return img
    
    
    
def get_mask(img):
    mask = find_mask(img, False)
    mask = grow_mask_outward(mask)
    return mask


def read_dicom(path):
    g = glob.glob(os.path.join(path, '*.dcm'))
    slices = [pydicom.read_file(s) for s in g]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices


def hu_clip_tensor(scan, upper, lower, min_max_norm=True):
    scan = torch.where(scan < lower, lower, scan)
    scan = torch.where(scan > upper, upper, scan)
    if min_max_norm:
        scan = min_max_normalize(scan)
        
    return scan


def hu_clip(scan, upper, lower, min_max_norm=True, zipped=False):
    scan = np.where(scan < lower, lower, scan)
    scan = np.where(scan > upper, upper, scan)

    if min_max_norm:
        scan = min_max_normalize(scan)
    
    if zipped:
        scan = (scan*255).astype(np.uint8).astype(np.float32) / 255
        
    return scan


def hu_window(scan, window_level=40, window_width=80, min_max_normalize=True):
    scan = scan.pixel_array.copy()
    window = [window_level-window_width//2, window_width//2-window_level]
    
    scan = np.where(scan < window[0], window[0], scan)
    scan = np.where(scan > window[1], window[1], scan)
    
    if min_max_normalize:
        scan = (((scan - scan.min()) / (scan.max() - scan.min())) * 255).astype(np.uint8).astype(np.float32) / 255

    return scan


def show_raw_pixel(slices):
    #讀出像素值並且儲存成numpy的格式
    image = hu_window(slices, window_level=0, window_width=1000,  show_hist=False)
#     plt.figure(figsize = (12,12))
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
    
    
    
# return  a region where cbct images aren't all black
def valid_slices(cbcts, cts):
    found_start = False
    start = 0
    end = -1
    
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
                                   