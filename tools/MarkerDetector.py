
import numpy as np
import cv2
from cv2 import aruco as aruco
import json
import os
from easydict import  EasyDict


from src import Camera


filename = "test_324/dataset/right/000000.png"
jsonname = "test_324/dataset/cameras.json"

pth_output_img  = "test_324/dataset/right/marker.png"
pth_output_json = "test_324/dataset/right/marker.json"

img = cv2.imread(filename)
with open(jsonname, "r") as f:
    tmp = EasyDict(json.load(f))
    cam1 = Camera.PinholeCamera(720, 1080)
    cam1.intrinsic.set_params(tmp.right.intrin.fx, tmp.right.intrin.fy, tmp.right.intrin.cx, tmp.right.intrin.cy)
    K = cam1.intrinsic._mat()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_tresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
img_left = np.where(img_tresh==0, img_gray, 0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(img_left, aruco_dict, parameters=parameters)
rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, K, np.zeros(5))

img_out = aruco.drawDetectedMarkers(img, corners)
for i in range(ids.size):
    id_ = ids[i]
    aruco.drawAxis(img_out, K, np.zeros(5), rvecs[i, :, :], tvecs[i, :, :], 0.05)
    cv2.putText(img_out, "Id: " + str(ids.flatten()), (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
cv2.namedWindow("", cv2.WINDOW_FREERATIO)
cv2.imshow("", img_out)
cv2.waitKey()
cv2.imwrite(pth_output_img, img_out)

output_data = EasyDict({})
output_data.rvecs = rvecs.tolist()
output_data.tvecs = tvecs.tolist()
with open(pth_output_json, "w") as f:
    json.dump(output_data, f)

