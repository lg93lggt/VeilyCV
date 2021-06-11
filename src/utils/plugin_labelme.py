
import numpy as np
import json
import cv2
import os
from icecream import ic
import glob 
import sys

sys.path.append("../..")
from src.utils.debug import debug_separator

@ debug_separator
def generate_mask(pth_label, color=1, type_output=np.float32):
    with open(pth_label, 'r') as f:
        data = json.load(f)
    pts = np.asarray(data['shapes'][0]['points'])
    size = (data["imageHeight"], data["imageWidth"])
    mask = np.zeros(size).astype(np.uint8)
    mask = cv2.fillPoly(mask, [pts.astype(int)], color)
    # plt.imshow(mask)
    # plt.show()
    return mask.astype(type_output)

@ debug_separator
def generate_masks(dir_label, color=1, type_output=np.float32, key=None, save_mask=False):
    [dir_parent, _] = os.path.split(dir_label)
    dir_mask = os.path.join(dir_parent, "mask")
    if save_mask:
        os.makedirs(dir_mask) if not os.path.exists(dir_mask) else None

    pthes_label = glob.glob(os.path.join(dir_label, "*.json"))
    pthes_label.sort(key=key) if key is not None else pthes_label.sort()

    masks = []
    for pth_label in pthes_label:
        [_, name_label] = os.path.split(pth_label)
        [prefix, _] = os.path.splitext(name_label)
        mask = generate_mask(pth_label, type_output=np.uint8, color=color)
        masks.append(mask)
        if save_mask:
            pth_output_mask = os.path.join(dir_mask, prefix+".jpg")
            cv2.imwrite(pth_output_mask, mask)
            ic(pth_output_mask)
    return np.array(masks)