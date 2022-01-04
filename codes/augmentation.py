import albumentations as albu
import cv2
import numpy as np

def image_resize(image, size = 256, inter = cv2.INTER_AREA, **kwargs):

        dim = None
        (h, w) = image.shape[:2]

        if size is None:
                return image

        # check to see if the width is None
        if h > w:
                r = size / float(h)
                dim = (int(w * r), size)
        else:
                r = size / float(w)
                dim = (size, int(h * r))

        resized = []
        for i in range(image.shape[-1]):
                tmp = image[..., i]
                tmp = cv2.resize(tmp, dim, interpolation = inter)
                resized.append(tmp)

        resized = np.stack(resized, axis=-1)

        return resized


def random_erasing(x, p=0.5, sl=0.02, sh=0.4, r1=0.3, **kwargs):
        import random
        import math

        W = x.shape[0]
        H = x.shape[1]
        S = W * H
        p1 = random.uniform(0, 1)
        r2 = 1 / r1

        if p1 >= p:
                return x
        else:
                while True:
                        se = random.uniform(sl, sh) * S
                        re = random.uniform(r1, r2)

                        he = math.sqrt(se * re)
                        we = math.sqrt(se / re)

                        xe = random.uniform(0, W)
                        ye = random.uniform(0, H)

                        if (xe+we <= W) and (ye+he <= H):

                                for _x in range(int(xe), int(xe+we)):
                                        for _y in range(int(ye), int(ye+he)):
                                                for c in range(0, 3):
                                                        x[_x, _y, c] = random.uniform(0, 255)

                                return x



                            

def intensity_augmentation():
        import augmentations
        img_transform = [
                    DicomWindowShift(window_width_mins=(1000),
                                                                 window_width_maxs=(1000),
                                                                 window_center_mins=(0),
                                                                 window_center_maxs=(0),
                                                                 min_max_normalize=False,
                                                                 p=1.0)
        ]
        return albu.Compose(img_transform, additional_targets={'image0': 'image'})




def get_training_augmentation():
        train_transform = [
                # ref: https://github.com/albumentations-team/albumentations/issues/640
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                albu.RandomCrop(384, 384, p=0.5),
                albu.LongestMaxSize(max_size=384, interpolation=cv2.INTER_LINEAR_EXACT, always_apply=True),
                albu.PadIfNeeded(min_height=384, min_width=384, always_apply=True, border_mode=0),
        ]
        return albu.Compose(train_transform)



def get_validation_augmentation():
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
                # ref: https://github.com/albumentations-team/albumentations/issues/640
                albu.LongestMaxSize(max_size=384, interpolation=cv2.INTER_LINEAR_EXACT, always_apply=True),
                albu.PadIfNeeded(384, 384, always_apply=True, border_mode=0)
        ]
        return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
        """Construct preprocessing transform
        
        Args:
                preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
                transform: albumentations.Compose
        
        """
        
        _transform = [
                                    DicomWindowShift(window_width_mins=(1000),
                                                                 window_width_maxs=(1000),
                                                                 window_center_mins=(0),
                                                                 window_center_maxs=(0),
                                                                 min_max_normalize=False,
                                                                 p=1.0)
        ]
        return albu.Compose(_transform)
