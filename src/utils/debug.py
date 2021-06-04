
from pathlib import Path
import cv2
import os
from icecream import ic
import time


def separator(func):
    def wrapper(*args, **kwargs):
        print("\n"+"*"*64)
        print("{}".format(func.__name__))
        print("*"*64)
        ts = time.time()
        ret = func(*args, **kwargs)
        te = time.time()
        use_time = te - ts
        print("*"*64)
        ic(use_time)
        print("*"*64+"\n")
        return ret
    return wrapper


def debug_vis(debug, dir_debug=None, name_img=None, img_out=None):
    if (dir_debug is None) or (img_out is None) or (name_img is None):
        return
    else:
        dir_debug = Path(dir_debug)
        dir_debug.mkdir(parents=True, exist_ok=True)
        pth_debug = Path(dir_debug, name_img)
        cv2.imwrite(str(pth_debug), img_out)
        ic(pth_debug.name)
        
    if debug:
        cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
        cv2.imshow("debug", img_out)
        cv2.waitKey(debug)
    else:
        pass
    return

if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    I = cv2.imread("/home/veily3/LIGAN/VeilyCV/test/test_508/test3/00001.jpg")
    M = cv2.imread("/home/veily3/LIGAN/VeilyCV/test/test_508/test3/00001.png", cv2.IMREAD_GRAYSCALE)
    A = np.loadtxt("/home/veily3/LIGAN/VeilyCV/test/test_508/test3/00001.txt", delimiter=",")

    X = cv2.Sobel(A, cv2.CV_64FC1, 1, 0)
    X = np.abs(X)
    X = (X / X.max() * 255).astype(np.uint8)


    Y = cv2.Sobel(A, cv2.CV_64FC1, 0, 1)
    Y = np.abs(Y)
    Y = (Y / Y.max() * 255).astype(np.uint8)

    XY_ = X+Y
    ofst = 50
    XY = np.zeros_like(XY_, dtype=np.uint8)
    XY[ofst:-ofst, ofst:-ofst] = XY_[ofst:-ofst, ofst:-ofst]

    K = cv2.bitwise_and(XY, XY, mask=M)
    _, K = cv2.threshold(K, 0, 1, cv2.THRESH_OTSU)

    K = cv2.erode(K, np.zeros((5), np.uint8), iterations=1)
    K = cv2.dilate(K, np.zeros((5), np.uint8), iterations=1)
    plt.subplot(2,2,1)
    plt.imshow(K)
    plt.show()


    cnts, _ = cv2.findContours(K, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  
    idx_max = np.argmax(np.asarray([len(i) for i in cnts]))
    #cnt_max = c
    I = cv2.drawContours(I, cnts[idx_max], -1, (0,0,255), 10) 
    plt.subplot(2,2,2)
    plt.imshow(I)
    plt.show()

    pts = cv2.approxPolyDP(	cnts[idx_max], 1., closed=False)

    I = cv2.drawContours(I, [pts], -1, (255,0,0), 1) 
    plt.subplot(2,2,4)
    plt.imshow(I)
    plt.show()
    print()