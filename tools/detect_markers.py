
import numpy as np
import cv2
from cv2 import aruco as aruco


import json
import os



def calc_depth_general_case(intrinsic1, intrinsic2, extrinsic1, extrinsic2, bbox1, bbox2, observer_idx=0):
    fx = intrinsic1._fx()
    fy = intrinsic1._fy()

    cx = intrinsic1._cx()
    cy = intrinsic1._cy()

    c1 = extrinsic1._c()
    c2 = extrinsic2._c()

    [x1, y1, w1, h1] = bbox1
    [x2, y2, w2, h2] = bbox2

    if (w1 == 0) or (w2 == 0) or (h1 == 0) or (h2 == 0):
        return -1
    else:
        s_w = w1 / w2
        s_h = h1 / h2
        
        if observer_idx == 0:
            s = (s_w + s_h) / 2

            z1_x = fx * (c1[0] - c2[0]) / \
                ((x2 - cx) * s - (x1 - cx))

            z1_y = fy * (c1[1] - c2[1]) / \
                ((y2 - cy) * s - (y1 - cy))
            z1 = (z1_y + z1_x) / 2
            return z1

        elif observer_idx == 1:
            s = (1/s_w + 1/s_h) / 2

            z1_x = fx * (c2[0] - c1[0]) / \
                ((x1 - cx) * s - (x2 - cx))

            z1_y = fy * (c2[1] - c1[1]) / \
                ((y1 - cy) * s - (y2 - cy))
            z1 = (z1_y + z1_x) / 2
            return z1


from src import Camera
from easydict import  EasyDict
        
import json
by_marker = 0
cams_marker = []    
cams_rltv = []
bboxs = []
img_outs = []
for idx in range(2):
    camera_m = EasyDict({})
    camera_r = EasyDict({})
    
    with open("test_320/camerapars/campars{:d}.json".format(idx), "r") as f:
        data = EasyDict(json.load(f))
    img = cv2.imread("test_320/img/dlt/0_{}.jpg".format( idx))

    K = np.array(data.intrin)[:3, :3]
    #dist = np.array(data["dist_coeff"])

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, K, np.zeros(5))

    img_out = aruco.drawDetectedMarkers(img, corners)
    for i in range(ids.size):
        aruco.drawAxis(img_out, K, np.zeros(5), rvecs[i, :, :], tvecs[i, :, :], 0.05)
        cv2.putText(img_out, "Id: " + str(ids.flatten()), (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
    cv2.namedWindow("", cv2.WINDOW_FREERATIO)
    cv2.imshow("", img_out)
    cv2.waitKey(1)
    img_outs.append(img_out)
    
    intrin  = Camera.Intrinsic(K)
    extrin  = Camera.Extrinsic()
    
    extrin.set_rtvec(rvecs[0], tvecs[0])
    camera_m.intrin = intrin
    camera_m.extrin = extrin
    cams_marker.append(camera_m)

    intrin  = Camera.Intrinsic(K)
    extrin  = Camera.Extrinsic()
    if idx == 0:
        e1 = np.array(data.extrin)
        extrin.set_matrix(np.eye(4))
    else:
        e2 = np.linalg.inv(e1) @ np.array(data.extrin)
        extrin.set_matrix(e2)

    camera_r.intrin = intrin
    camera_r.extrin = extrin
    cams_rltv.append(camera_r)
    print()


    with open("test_320/img/{}_{}.json".format(idx, idx)) as f :
        tmp = f.read()
        data = EasyDict(json.loads(tmp))
        box = np.array(data.shapes[0]["points"]).astype(np.int)
        print()
        pts = cv2.rectangle(img, tuple(box[0]), tuple(box[1]), color=(0, 0, 255))
        [w, h] = box[1] - box[0]
        cv2.imshow("", pts)
        cv2.waitKey()
        bbox = [box[0, 0], box[1, 1], w, h]  
        bboxs.append(bbox)

ex = np.linalg.inv(cams_marker[0].extrin._mat_4x4()) @ np.linalg.inv(cams_marker[1].extrin._mat_4x4())
idcam1 = Camera.PinholeCamera(480, 640, cams_marker[0].intrin.mat)
idcam1.set_extrinsic(Camera.Extrinsic())
idcam2 = Camera.PinholeCamera(480, 640, cams_marker[1].intrin.mat)
idcam2.set_extrinsic(Camera.Extrinsic(ex[:3]))
idcam3 = Camera.PinholeCamera(480, 640, cams_marker[0].intrin.mat)
idcam3.render_options.coord.length = 0.1
idcam3.set_extrinsic(Camera.Extrinsic())
idcam4 = Camera.PinholeCamera(480, 640, cams_marker[1].intrin.mat)
idcam4.render_options.coord.length = 0.2
idcam4.set_extrinsic(cams_rltv[1].extrin)

from src.CommonGeometricShapes import Cuboid

import open3d as o3d

vis = o3d.visualization.Visualizer()
vis.create_window("o3d", width=640, height=480)
opt = vis.get_render_option()
opt.background_color = np.zeros(3)


cube = Cuboid.Cuboid(0.2, 0.2, 0.2, center=np.ones(3)*0.1)
cube2 = Cuboid.Cuboid(0.12, 0.12, 0.12, center=np.array([0, 0, 0.06]))
cube2.vertexes[:, 2] = -cube2.vertexes[:, 2]
cube2._refresh()

cube.transform3d(e1)
cube.draw(vis)

cube2.transform3d(cams_marker[0].extrin._mat_4x4())
cube2.draw(vis)

idcam1.view(vis)
img1   = vis.capture_screen_float_buffer()
img1   = (np.asarray(img1  ) *  255).astype(np.uint8)
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
idcam4.view(vis)
#vis.run()
img2   = vis.capture_screen_float_buffer()
img2   = (np.asarray(img2  ) *  255).astype(np.uint8)
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
#vis.run()
imgs = [img1, img2]

for idx in range(2):
    img = cv2.imread("test_320/img/dlt/0_{}.jpg".format( idx))
    cv2.imshow(str(idx), cv2.addWeighted(img_outs[idx],0.5, imgs[idx],0.5, 0))
    cv2.waitKey()

ret1 = calc_depth_general_case(cams_marker[0].intrin, cams_marker[1].intrin, Camera.Extrinsic(), Camera.Extrinsic(ex[:3]), bboxs[0], bboxs[1], 0)
ret2 = calc_depth_general_case(cams_marker[0].intrin, cams_marker[1].intrin, Camera.Extrinsic(), Camera.Extrinsic(ex[:3]), bboxs[0], bboxs[1], 1)

ret3 = calc_depth_general_case(cams_rltv[0].intrin, cams_rltv[1].intrin, cams_rltv[0].extrin, cams_rltv[1].extrin, bboxs[0], bboxs[1], 0)
ret4 = calc_depth_general_case(cams_rltv[0].intrin, cams_rltv[1].intrin, cams_rltv[0].extrin, cams_rltv[1].extrin, bboxs[0], bboxs[1], 1)
print()





