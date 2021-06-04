

import cv2
import os
import glob
import numpy as np
from icecream import ic

def rename_images(dir_folder, suffix=".jpg", key=None, shape=None):
    pthes_img = glob.glob(os.path.join(dir_folder, "*"+suffix))
    pthes_img.sort() if key is None else pthes_img.sort(key=key)

    rets = np.zeros(len(pthes_img)).astype(bool)
    for [idx_pth, pth_img] in enumerate(pthes_img):
        img = cv2.imread(pth_img)
        if shape is None:
            pass
        else:
            if (img.shape[0] == shape[1]) and (img.shape[1] == shape[0]):
                img = np.rot90(img)
                # ic(img.shape)
        pth_output = os.path.join(dir_folder, "{:0>6d}".format(idx_pth+1) + suffix)
        ret = cv2.imwrite(filename=pth_output, img=img)
        rets[idx_pth] = ret
        ic(ret, pth_output)
    
    if rets.all():
        for pth_img in pthes_img:
            os.remove(pth_img)
    else:
        pass

if __name__ == '__main__':
    dir_folder = "/home/veily3/LIGAN/VeilyCV/test/test_507/fit/test1/raw"
    rename_images(dir_folder, shape=(1440, 1080))