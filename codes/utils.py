import glob
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt


def read_dicom(path):
    g = glob.glob(os.path.join(path, '*.dcm'))
    slices = [pydicom.read_file(s) for s in g]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices


def hu_clip(scan, upper, lower, min_max_normalize=True):
    scan = np.where(scan < lower, lower, scan)
    scan = np.where(scan > upper, upper, scan)

    if min_max_normalize:
        scan = (((scan - scan.min()) / (scan.max() - scan.min())) * 255).astype(np.uint8).astype(np.float32) / 255
        
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
                                   