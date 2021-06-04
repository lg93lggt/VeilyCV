
import cv2
import numpy as np


def make_gaussian_noise(shape_image, mean, std):
    """
        shape_image:
        mean:
        std:
    """
    if   len(shape_image)==2:
        [h, w] = shape_image
        img_noise = cv2.randn(np.zeros((h, w)), mean, std)
        return img_noise

    elif len(shape_image)==3:
        [h, w, c] = shape_image
        r = cv2.randn(np.zeros((h, w)), mean, std)
        g = cv2.randn(np.zeros((h, w)), mean, std)
        b = cv2.randn(np.zeros((h, w)), mean, std)

        if c==1:
            img_noise = r
            return img_noise

        elif c==3:
            img_noise = cv2.merge([b, g, r])
            return img_noise

        else:
            raise IndexError
    else:
        raise IndexError
    

def add_gaussian_noise(img_src, mean, std):
    """
        Add 1 channel gaussian noise to source image. 
        input args:
            shape_image:
            mean:
            std:
    """
    n_dims = len(img_src.shape)
    if   n_dims==2:
        [h, w] = img_src.shape
        img_noise = make_gaussian_noise((h, w), mean, std)
        dst = img_src + img_noise

    elif n_dims==3:
        [h, w, c] = img_src.shape
        img_noise = make_gaussian_noise((h, w), mean, std)
        img_noise = np.expand_dims(img_noise, axis=2).repeat(repeats=c, axis=2)
        dst = img_src + img_noise

    else:
        raise IndexError

    dst[np.where(dst > 255)] = 255
    dst[np.where(dst < 0  )] = 0
    dst = dst.astype(np.uint8)
    return dst


def add_gaussian_noise_multi_channels(img_src, mean, std):
    """
        Add n channels (which is same as source image channels) gaussian noise to source image.
    """
    [h, w, c] = img_src.shape
    img_noise = make_gaussian_noise(img_src.shape, mean, std)
    
    dst = img_src + img_noise
    dst[np.where(dst > 255)] = 255
    dst[np.where(dst < 0  )] = 0
    dst = dst.astype(np.uint8)
    return dst
