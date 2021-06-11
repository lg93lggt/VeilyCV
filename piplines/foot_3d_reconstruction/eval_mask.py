
from os import path
import cv2
import numpy as np
from pathlib import Path

dir_root = Path("/home/veily3/LIGAN/VeilyCV/test/test_517/LG_video")
dir_raw = Path(dir_root, "raw")
dir_mask = Path(dir_root, "mask")

for pth_img in sorted(dir_raw.glob("*.*")):
    pth_mask = dir_mask / pth_img.name
    mask = cv2.imread(str(pth_mask), cv2.IMREAD_GRAYSCALE)
    img  = cv2.imread(str(pth_img))
    mask = cv2.merge((mask, np.zeros_like(mask, dtype=np.uint8), np.zeros_like(mask, dtype=np.uint8)))
    img_show = cv2.addWeighted(img, 0.8, mask, 0.3, 0)
    cv2.imwrite(str(dir_root / "show" / pth_img.name), img_show)
    cv2.imshow("", img_show)
    cv2.waitKey(100)
    
