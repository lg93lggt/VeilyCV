
import cv2
from cv2 import aruco as aruco
import  numpy as np
import os

# Load the predefined dictionary
dictionary = aruco.Dictionary_get(aruco.DICT_4X4_250)
[H, W] = [256, 256]
dx = 32
marker_size = 256//8*6
for idx in range(80):
# Generate the marker
    img_marker = np.full((H, W, 3), 255, dtype=np.uint8) 
    cv2.rectangle(img_marker, (0, 0), (H-1, W-1), (0,0,255))
    marker = aruco.drawMarker(dictionary, idx, marker_size, img_marker, 1);
    img_marker[dx:marker_size+dx, dx:marker_size+dx, :] = cv2.merge([marker, marker, marker])
    dir_root = "./marker_base"
    dir_sub  = os.path.join(dir_root, "DICT_4X4_250_outlier")
    os.makedirs (dir_sub) if not os.path.exists(dir_sub) else 1
    cv2.imshow("", img_marker)
    cv2.waitKey()
    pth_out = os.path .join(dir_sub, "marker_{:0>2d}.png".format(idx))
    cv2.imwrite(pth_out , img_marker)
    print(pth_out)
