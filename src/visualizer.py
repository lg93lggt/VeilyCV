
from typing import *
import cv2
import numpy as np
from easydict import EasyDict
import sys
try:
    import geometries as geo
except :    
    #sys.path.append("./src")
    import src.geometries as geo
#from src.core import Ellipse2d

def to_plot(point2d, is_homo=False) -> np.ndarray:
    if is_homo:
        p2d = tuple(np.round(point2d[:2]).flatten().astype(np.int).tolist())
    else:
        p2d = tuple(np.round(point2d).flatten().astype(np.int).tolist())
    
    return p2d
    
def draw_points2d(img, points2d_chosen: np.ndarray, radius: int=5, color: Tuple[int]=(0, 127, 0)):
    for point2d in points2d_chosen:
        cv2.circle(img, to_plot(point2d), radius, color, 1)
    return

def draw_points3d(
        img: np.ndarray, 
        points3d: np.ndarray, 
        rtvec: np.ndarray, camera_pars: Dict, 
        radius=1,
        color: Tuple[int]=(255, 127, 0)
    ):

    if camera_pars is None:
        return
    else:
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        points2d = geo.project_points3d_to_2d(rtvec, M, points3d)
        draw_points2d(img, points2d, radius=radius, color=color)
        return

def draw_points2d_with_texts(img, points2d_chosen: np.ndarray, indexes: np.ndarray, radius: int=5, color: Tuple[int]=(0, 127, 255)):
    if points2d_chosen is None or indexes is None :
        return
    for [i_point, point2d] in enumerate(points2d_chosen):
        cv2.circle(img, to_plot(point2d), radius, color, 1, 0)
        off_set = 5
        text = "{}".format(indexes[i_point] + 1)
        cv2.putText(img, text, to_plot(point2d + off_set), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=color)
    return
    
def draw_points3d_with_texts(img, points3d: np.ndarray, rtvec: np.ndarray, camera_pars, radius=1, color: Tuple[int]=(255, 127, 0)):
    if camera_pars is None:
        return
    else:
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        points2d = geo.project_points3d_to_2d(rtvec, M, points3d)
        draw_points2d(img, points2d, radius=radius, color=color)
        n_points = points2d.shape[0]
        for i_point in range(n_points):
            point2d = points2d[i_point]
            off_set = -5
            text = "{}".format(i_point + 1)
            cv2.putText(img, text, to_plot(point2d + off_set), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=color)
        return

def draw_backbone3d(
        img: np.ndarray, 
        points3d_backbone: np.ndarray, 
        backbone_lines: np.ndarray,
        rtvec: np.ndarray, 
        camera_pars, 
        color: Tuple[int]=(255, 255, 128), 
        width_line = 1
    ):

    if (points3d_backbone is None) or (backbone_lines is None) or (camera_pars is None):
        return
    else:
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        points2d = geo.project_points3d_to_2d(rtvec, M, points3d_backbone)
        n_points3d = points2d.shape[0]
        n_lines  = backbone_lines.size // 2
        for i_line in range(n_lines):
            idx1 = backbone_lines[i_line][0]
            idx2 = backbone_lines[i_line][1]
            # check
            if (idx1 < n_points3d) and (idx2 < n_points3d):
                cv2.line(img, to_plot(points2d[idx1]), to_plot(points2d[idx2]), color, width_line)
            else:
                continue
        return

def draw_backbone2d(
        img: np.ndarray, 
        points2d: np.ndarray, 
        indexes3d: np.ndarray,
        backbone_lines: np.ndarray,
        color: Tuple[int]=(255, 255, 128), 
        width_line = 1
    ):

    if (points2d is None) or (indexes3d is None) or (backbone_lines is None):
        return
    n_points2d = points2d.shape[0]
    n_lines  = backbone_lines.size // 2
    for i_line in range(n_lines):
        idx1 = np.where(indexes3d==backbone_lines[i_line][0])[0]
        idx2 = np.where(indexes3d==backbone_lines[i_line][1])[0]
        # check
        if (idx1.size > 0) and (idx2.size > 0):
            if (idx1[0] < n_points2d) and (idx2[0] < n_points2d) and (backbone_lines[i_line][0] in indexes3d) and (backbone_lines[i_line][1] in indexes3d):
                cv2.line(img, to_plot(points2d[idx1]), to_plot(points2d[idx2]), color, width_line)
            else:
                continue
        else:
            continue
    return

def draw_axes3d(img, camera_pars: Dict, rtvec: np.ndarray=np.zeros(6), unit_length: float=0.1, width_line: int=1):
    if camera_pars is None:
        return
    else:
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        p2ds = geo.project_points3d_to_2d(rtvec, M, np.array([[0, 0, 0], [unit_length, 0, 0], [0, unit_length, 0], [0, 0, unit_length]]))
        cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[1]), (0, 0, 255), width_line)
        cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[2]), (0, 255, 0), width_line)
        cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[3]), (255, 0, 0), width_line)
        return
    
def draw_triangle2d(img, points2d_tri, color):
    p2ds = []
    for i in [0, 1, 2]:
        p2d = to_plot(points2d_tri[i])
        p2ds.append(p2d)

    cv2.line(img, p2ds[0], p2ds[1], color, 1)
    cv2.line(img, p2ds[1], p2ds[2], color, 1)
    cv2.line(img, p2ds[2], p2ds[0], color, 1)
    return

def draw_model3d(img, model, rtvec, camera_pars, color=(0, 255, 0)):
    if camera_pars is None:
        return
    else:
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        points3d_model = []
        points2d_model = []
        for tri in (model):
            points3d_tri = []
            points2d_tri = []
            for point3d in tri:
                p3d = np.array([[point3d[0], point3d[1], point3d[2]]]) / 1000
                points3d_tri.append(p3d)
                p2d = geo.project_points3d_to_2d(
                    rtvec, 
                    M,
                    p3d
                )
                points2d_tri.append(p2d)
            points3d_model.append(points3d_tri)
            points2d_model.append(points2d_tri)
            draw_triangle2d(img, points2d_tri, color)
        return

def draw_model3d_mask(img, model, rtvec, camera_pars, color=(0, 255, 0), alpha=0.5):
    mask = np.zeros(img.shape[:2], np.uint8)

    if camera_pars is None:
        return
    else:
        M = camera_pars["intrin"] @ camera_pars["extrin"]
        points2d_n_tris = []
        for tri in (model):
            points2d_tri = []
            for point3d in tri:
                p3d = np.array([[point3d[0], point3d[1], point3d[2]]]) / 1000
                p2d = geo.project_points3d_to_2d(
                    rtvec, 
                    M,
                    p3d
                )
                points2d_tri.append(to_plot(p2d))
            points2d_n_tris.append(points2d_tri) 
            cv2.fillPoly(mask, np.array([points2d_tri]), color=1)
            # cv2.imshow("", img)
            # cv2.waitKey(0)   
    
        img[np.where(mask!=0)] =  np.round(img[np.where(mask!=0)] * (1 - alpha) + np.array(color) * alpha)
        return

# def draw_backbone2d_ellipse(img, points2d, color=(255, 255, 255), width_line=1):
#     pt_head     = points2d[0]
#     pts_ellipse = points2d[1:]
#     e = Ellipse2d.Ellipse2d()
#     e._set_by_5points2d(pts_ellipse)
#     e.draw(img, color=color, thickness=width_line)
#     for pt in pts_ellipse:
#         cv2.line(img, to_plot(pt), to_plot(pt_head), color, width_line)
#     return

def draw_log(log_data):
    import matplotlib.pyplot as plt
    log_theta = log_data[:, :-1]
    log_loss  = log_data[:, -1]
    plt.plot(range(log_loss.size), log_loss)
    plt.show()
    return
