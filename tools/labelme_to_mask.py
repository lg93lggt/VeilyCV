
from pathlib import Path
from typing import Union
import numpy as np
import cv2
import json
import glob
import os
from easydict import EasyDict

def convert_label_to_mask(pth_json: Union[str, Path]) -> np.ndarray:
    with open(pth_json, 'r') as fp:
        data = EasyDict(json.load(fp))
    pts = np.asarray(data["shapes"][0]["points"])
    size = (data.imageHeight, data.imageWidth)
    mask = np.zeros(size).astype(np.uint8)
    mask = cv2.fillPoly(mask, [pts.astype(int)], 1)
    return mask

def convert_labels_to_maskes(dir_json, dir_output_mask, suffix_output=".jpg"):
    if not os.path.exists(dir_output_mask):
        os.makedirs(dir_output_mask)

    pthes_json = glob.glob(dir_json+"/*.json")
    pthes_json.sort()
    for pth_json in pthes_json:
        [_, name_json] = os.path.split(pth_json)
        [prefix, _] = os.path.splitext(name_json)
        pth_output_mask = os.path.join(dir_output_mask, prefix+suffix_output)

        mask = convert_label_to_mask(pth_json)
        cv2.imwrite(pth_output_mask, mask)
        print("Save mask:", pth_output_mask)
    return


if __name__ == '__main__':
    convert_labels_to_maskes("test_428/labels", "test_428/masks")