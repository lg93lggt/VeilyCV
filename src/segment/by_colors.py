
import numpy as np
import cv2
import json
import os
import glob
import open3d as o3d
from matplotlib import pyplot as plt
from easydict import EasyDict
from src import Camera
from src import geometries
from icecream import ic



blue_hsv = np.array([100, 190, 160])
green_hsv  = np.array([50, 150, 100])

green_lab = np.array([165,  90, 150])
blue_lab  = np.array([ 90, 135,  90])
red_lab   = np.array([ 95, 165, 140])

def segment_by_colors(img, colors, delta=40, repeat=2, debug=0):
    delta  = delta
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    if debug:
        cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
        cv2.imshow("debug", img_cvt)
        cv2.waitKey(debug)

    bnds_lower = colors - delta
    bnds_upper = colors + delta

    imgs_roi = []
    maskes   = []
    for idx_bnd in range(len(colors)):
        lb = bnds_lower[idx_bnd]
        ub = bnds_upper[idx_bnd]

        mask = cv2.inRange(img_cvt, lowerb=lb, upperb=ub)
        for _ in range(repeat):
            mask = cv2.erode(mask, kernel)
            mask = cv2.dilate(mask, kernel)
        mask = cv2.dilate(mask, kernel)
        maskes.append(mask)
    
        roi = cv2.bitwise_and(img, img, mask=mask)
        imgs_roi.append(roi)

    if debug:
        for [idx, mask] in enumerate(maskes):
            cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
            cv2.imshow("debug", imgs_roi[idx])
            cv2.waitKey(debug)   
    return imgs_roi


def segment_by_colors_inv(img, colors, delta=40, repeat=2, inv_color=True, debug=0):
    if inv_color:
        colors = np.flip(colors, axis=0)

    delta  = delta
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    if debug:
        cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
        cv2.imshow("debug", img_cvt)
        cv2.waitKey(debug)

    bnds_lower = colors - delta
    bnds_upper = colors + delta

    imgs_roi = []
    maskes   = []
    for idx_bnd in range(len(colors)):
        lb = bnds_lower[idx_bnd]
        ub = bnds_upper[idx_bnd]

        mask = cv2.inRange(img_cvt, lowerb=lb, upperb=ub)
        for _ in range(repeat):
            mask = cv2.erode(mask, kernel)
            mask = cv2.dilate(mask, kernel)
        mask = cv2.bitwise_not(mask)
        mask = cv2.dilate(mask, kernel)
        maskes.append(mask)
        
        roi = cv2.bitwise_and(img, img, mask=mask)
        imgs_roi.append(roi)

    if debug:
        for [idx, mask] in enumerate(maskes):
            cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
            cv2.imshow("debug", imgs_roi[idx])
            cv2.waitKey(debug)   
    return imgs_roi

