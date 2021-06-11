

import cv2
import os
import glob
import numpy as np
from pathlib import  Path
from icecream import ic

def rename_images(dir_input, dir_output, suffix=".jpg", key=None, shape=None, downsample=1):
    pthes_img = glob.glob(os.path.join(dir_input, "*.*"))
    pthes_img.sort() if key is None else pthes_img.sort(key=key)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    ic(dir_output)
    # rets = np.zeros(len(pthes_img)).astype(bool)
    for [idx_pth, pth_img] in enumerate(pthes_img):
        if idx_pth % downsample != 0:
            continue
        else:
            pass

        img = cv2.imread(pth_img)

        if idx_pth == 0:
            shape = img.shape[:2] if (shape is None) else shape
        else:
            if (img.shape[0] == shape[1]) and (img.shape[1] == shape[0]):
                img = np.rot90(img)
                # ic(img.shape)
        pth_output = Path(dir_output, "{:0>6d}".format(idx_pth+1) + suffix)
        ret = cv2.imwrite(filename=str(pth_output), img=img)
        # rets[idx_pth] = ret
        ic(pth_output.name)
    return
    
    # if rets.all():
    #     for pth_img in pthes_img:
    #         os.remove(pth_img)
    # else:
    #     pass

if __name__ == '__main__':
    dir_folder = "/home/veily3/LIGAN/VeilyCV/test/test_507/fit/test1/raw"
    rename_images(dir_folder, shape=(1440, 1080))