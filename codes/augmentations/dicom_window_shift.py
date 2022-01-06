"""
reference: https://www.kaggle.com/pestipeti/custom-albumentation-dicomwindowshift"""

import numpy as np
import random
from albumentations.core.transforms_interface import ImageOnlyTransform


def apply_window(image, center, width):
    image = image.copy()

    min_value = center - width // 2
    max_value = center + width // 2

    image[image < min_value] = min_value
    image[image > max_value] = max_value

    return image


def dicom_window_shift(img, windows, min_max_normalize=True):
    channels = len(windows)
    image = np.zeros((img.shape[0], img.shape[1], channels))

    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], channels, axis=2)

    for i in range(channels):
        ch = apply_window(img[:, :, i], windows[i][0], windows[i][1])

        if min_max_normalize:
            image[:, :, i] = (((ch - ch.min()) / (ch.max() - ch.min())) * 255).astype(np.uint8).astype(np.float32) / 255
        else:
            image[:, :, i] = ch

    return image


class DicomWindowShift(ImageOnlyTransform):
    """Randomly shift the DICOM window (per channel) between min and max values.
    
    Note: It won't work for preprocessed png or jpg images. Please use the dicom's HU values
    (rescaled width slope/intercept!)
    
    Args:
        window_width_mins (sequence of int): minimun window width per channel
        window_width_maxs (sequence of int): maximum window width per channel
        window_center_mins (sequence of int): minimum value for window center per channel
        window_center_maxs (sequence of int): maximum value for window center per channel
        min_max_normalize: (bool) Apply min-max normalization
    Targets:
        image
    Image types:
        uint8 (shape: HxW | HxWxC)
    """
    def __init__(
            self,
            window_width_mins=(80, 200, 380),
            window_width_maxs=(80, 200, 380),
            window_center_mins=(40, 80, 40),
            window_center_maxs=(40, 80, 40),
            min_max_normalize=True,
            always_apply=False,
            p=0.5,
    ):
        super(DicomWindowShift, self).__init__(always_apply, p)
        self.window_width_mins = window_width_mins
        self.window_width_maxs = window_width_maxs
        self.window_center_mins = window_center_mins
        self.window_center_maxs = window_center_maxs
        self.min_max_normalize = min_max_normalize

        assert len(self.window_width_mins) ==  len(self.window_width_maxs)
        assert len(self.window_center_mins) == len(self.window_center_maxs)
        
        self.channels = len(self.window_width_mins)


    def apply(self, image, windows=(), min_max_normalize=True, **params):
        return dicom_window_shift(image, windows, min_max_normalize)

    
    def get_params_dependent_on_targets(self, params):
        windows = []

        for i in range(self.channels):
            window_width = random.randint(self.window_width_mins[i], self.window_width_maxs[i])
            window_center = random.randint(self.window_center_mins[i], self.window_center_maxs[i])

            windows.append([window_center, window_width])

        return {"windows": windows, "min_max_normalize": self.min_max_normalize}
    

    @property
    def targets_as_params(self):
        return ["image"]
    

    def get_transform_init_args_names(self):
        return "window_width_mins", "window_width_maxs", "window_center_mins", "window_center_maxs", "min_max_normalize"

    