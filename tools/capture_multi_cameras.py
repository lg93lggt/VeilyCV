
import os
import cv2
import numpy as np
import time
from cv2 import aruco as aruco  



def images_capture(ids_camera, dir_output="./data", fps=5):

    delay  = False
    stream = False
    n_cams = len(ids_camera)

    cams = []
    for id_cam in ids_camera:
        cam = cv2.VideoCapture(id_cam)
        cams.append(cam)  

    is_all_opened = True
    for cam in cams:
        is_all_opened = is_all_opened or cam.isOpened()
    
    if is_all_opened:
        for [idx, _] in enumerate(cams):
            name_win = "{}".format(ids_camera[idx])
            cv2.namedWindow(name_win, cv2.WINDOW_NORMAL)

        cnt = 0
        do_exit = False
        while not do_exit:
            dt = 0
            t1 = time.time()
            frames = []

            for [idx, cam] in enumerate(cams):
                [ret, frame] = cam.read()
                if ret:
                    name_win = "{}".format(ids_camera[idx])
                    cv2.imshow(name_win, frame)
                    key = cv2.waitKey(1)

                if key == ord("q"):
                    do_exit = True

            if key == ord("c"):
                for [idx, cam] in enumerate(cams):
                    if not os.path.exists(os.path.join(dir_output,"{}".format(idx))):
                        os.makedirs(os.path.join(dir_output,"{}".format(idx)))
                    [ret, frame] = cam.read()
                    name_img = "{:0>6d}.jpg".format(cnt)
                    pth_img = os.path.join(dir_output,"{}".format(idx), name_img)
                    cv2.imwrite(pth_img, frame)
                    print(pth_img)
                cnt += 1

            if key == ord("s"):
                stream = True
            if stream:
                time.sleep(1/fps)
                for [idx, cam] in enumerate(cams):
                    if not os.path.exists(os.path.join(dir_output,"{}".format(idx))):
                        os.makedirs(os.path.join(dir_output,"{}".format(idx)))
                    [ret, frame] = cam.read()
                    name_img = "{:0>6d}.jpg".format(cnt)
                    pth_img = os.path.join(dir_output,"{}".format(idx), name_img)
                    cv2.imwrite(pth_img, frame)
                    print(pth_img)
                cnt += 1

            if key == ord("d"):
                t0 = time.time()
                delay = True
            if delay:
                print(t1-t0)
                if int(t1 - t0) % 10 == 0 and int(t1 - t0) != dt:
                    dt = int(t1 - t0)
                    for [idx, cam] in enumerate(cams):
                        if not os.path.exists(os.path.join(dir_output,"{}".format(idx))):
                            os.makedirs(os.path.join(dir_output,"{}".format(idx)))
                        [ret, frame] = cam.read()
                        name_img = "{:0>6d}.jpg".format(cnt)
                        pth_img = os.path.join(dir_output,"{}".format(idx), name_img)
                        cv2.imwrite(pth_img, frame)
                        print(pth_img)
                    cnt += 1
                else:
                    continue


def images_capture_markers(ids_camera, dir_output="./data", fps=5):

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters =  aruco.DetectorParameters_create()

    delay  = False
    stream = False
    n_cams = len(ids_camera)

    cams = []
    for id_cam in ids_camera:
        cam = cv2.VideoCapture(id_cam)
        cams.append(cam)  

    is_all_opened = True
    for cam in cams:
        is_all_opened = is_all_opened or cam.isOpened()
    
    if is_all_opened:
        for [idx, _] in enumerate(cams):
            name_win = "{}".format(ids_camera[idx])
            cv2.namedWindow(name_win, cv2.WINDOW_NORMAL)

        cnt = 0
        do_exit = False
        while not do_exit:
            dt = 0
            t1 = time.time()
            frames = []

            for [idx, cam] in enumerate(cams):
                [ret, frame] = cam.read()
                if ret:
                    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #ret, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)
                    frame_succ = aruco.drawDetectedMarkers(frame.copy(), corners)
                    if ids is not None:
                        for i in range(ids.size):
                            cv2.putText(frame_succ, "Id: " + str(ids.flatten()), (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

                    name_win = "{}".format(ids_camera[idx])
                    cv2.imshow(name_win, frame_succ)
                    key = cv2.waitKey(1)

                if key == ord("q"):
                    do_exit = True

            if key == ord("a"):
                for [idx, cam] in enumerate(cams):
                    if not os.path.exists(os.path.join(dir_output,"{}".format(idx))):
                        os.makedirs(os.path.join(dir_output,"{}".format(idx)))
                    [ret, frame] = cam.read()
                    name_img = "{:0>6d}.jpg".format(cnt)
                    pth_img = os.path.join(dir_output,"{}".format(idx), name_img)
                    cv2.imwrite(pth_img, frame)
                    print(pth_img)
                cnt += 1

            if key == ord("s"):
                stream = True
            if stream:
                time.sleep(1/fps)
                for [idx, cam] in enumerate(cams):
                    if not os.path.exists(os.path.join(dir_output,"{}".format(idx))):
                        os.makedirs(os.path.join(dir_output,"{}".format(idx)))
                    [ret, frame] = cam.read()
                    name_img = "{:0>6d}.jpg".format(cnt)
                    pth_img = os.path.join(dir_output,"{}".format(idx), name_img)
                    cv2.imwrite(pth_img, frame)
                    print(pth_img)
                cnt += 1

            if key == ord("d"):
                t0 = time.time()
                delay = True
            if delay:
                print(t1-t0)
                if int(t1 - t0) % 10 == 0 and int(t1 - t0) != dt:
                    dt = int(t1 - t0)
                    for [idx, cam] in enumerate(cams):
                        if not os.path.exists(os.path.join(dir_output,"{}".format(idx))):
                            os.makedirs(os.path.join(dir_output,"{}".format(idx)))
                        [ret, frame] = cam.read()
                        name_img = "{:0>6d}.jpg".format(cnt)
                        pth_img = os.path.join(dir_output,"{}".format(idx), name_img)
                        cv2.imwrite(pth_img, frame)
                        print(pth_img)
                    cnt += 1
                else:
                    continue



if __name__ == '__main__':
    
    list_cams_idx = [0, 2]
    images_capture(list_cams_idx, "./test_407")
    #images_capture_markers(list_cams_idx, "./test_331/data")
    # 11110000 11110000 11110000
    # 11110000 11110000
    print()