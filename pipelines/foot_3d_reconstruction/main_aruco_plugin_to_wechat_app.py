
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from cv2 import aruco as aruco
from icecream import ic
from easydict import  EasyDict

import calibrate_camera_aruco
import reconstruction_aruco
import registration_aruco
import open3d as o3d

dir_dnn = Path("/home/veily3/LIGAN/detectron")
sys.path.append(str(dir_dnn))
import predict_foot_PR

sys.path.append("../..")
from tools import video2images, rename_images


zc1 = np.array([-np.inf, 0])
zc2 = np.array([0.05, np.inf,])
zc3 = np.array([0, np.inf])


def main_vid(dir_output_project, dir_output_obj, name_output_obj, known_box_constraint="a4", unitized_length="m", **kwargs):
    """
        @dir_output_project:   工程文件夹
        @dir_output_obj:       输出文件夹
        @name_output_obj:      输出名称-要带后缀.obj
        [@pth_video:           视频路径
        [@dir_image:           图片文件夹]]
        @known_box_constraint: "a4"/"a3"/None
        @unitized_length:      单位长度 "m"/"cm"/"mm

    """

    """
    1. from pth_video
    2. from dir_image
    """
    if "dir_image" not in kwargs.keys():
        dir_image = None
    else:
        dir_image = kwargs["dir_image"]

    if "pth_video" not in kwargs.keys():
        pth_video = None
    else:
        pth_video   = Path(kwargs["pth_video"])

    dir_project = Path(dir_output_project)
    if   (pth_video is not None) and (dir_image is None):
        dir_output_raw = Path(dir_output_project, "raw")
        video2images.viedo2images(pth_video=pth_video, dir_output=dir_output_raw, downsample=20)
    elif (dir_image is not None) and (pth_video is None):
        dir_output_raw = Path(dir_output_project, "raw")
        rename_images.rename_images(dir_image, dir_output_raw, shape=None, suffix=".png")
    else:
        raise ValueError("Choose one of dir_image/pth_video.")

    """
    DNN
    """
    dnn_yaml_path       = Path(dir_dnn, "detectron/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
    dnn_model_path      = Path(dir_dnn, "output/footmask_PR", "1_model_final.pth")
    dnn_pthes_input_img = sorted(Path(dir_project, "raw").glob('*.*'))
    dnn_output_path     = Path(dir_project,'mask')
    predict_foot_PR.main(
        yaml_path=str(dnn_yaml_path), 
        model_path=str(dnn_model_path), 
        pthes_input_img=dnn_pthes_input_img, 
        output_path=str(dnn_output_path),
        p=0.5
    )


    """
    Reconstruction
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    if   known_box_constraint == "a3":      
        board_aruco = aruco.GridBoard_create(markersX=13, markersY=9, markerLength=0.026, markerSeparation=0.0045, dictionary=aruco_dict)
    elif known_box_constraint == "a4":
        board_aruco = aruco.GridBoard_create(markersX=6, markersY=9, markerLength=0.026, markerSeparation=0.0045, dictionary=aruco_dict)
    else:
        raise ValueError("known_box_constraint in a3/a4")

    calibrate_camera_aruco.main(
        board_aruco=board_aruco,
        dir_input=dir_project, 
        debug=False,
    )
    mesh = reconstruction_aruco.main(
        dir_fit_root=dir_project, 
        dir_calib_root=dir_project, 
        debug=False, 
        size_img=None, 
        z_constraint=zc3,
        known_box_constraint=known_box_constraint,
        prop=0.8
    )

    length_ratio_optional = EasyDict({"m": 1, "cm": 100, "mm": 1000})
    mesh.scale(length_ratio_optional[unitized_length], center=mesh.get_center()) # cm? mm?
    bbox = mesh.get_axis_aligned_bounding_box()
    [length_foot, width_foot, height_foot] = bbox.get_extent()
    ic(length_foot, width_foot, height_foot, unitized_length)
    
    # debug
    #o3d.visualization.draw_geometries([mesh, bbox])

    """
    IO
    """
    dir_output_obj = Path(dir_output_obj)
    dir_output_obj.mkdir(parents=True, exist_ok=True)
    pth_output = Path(dir_output_obj, name_output_obj)
    o3d.io.write_triangle_mesh(str(pth_output), mesh, write_vertex_colors=False, write_vertex_normals=False, print_progress=True)
    ic(pth_output)
    return length_foot, width_foot


if __name__ == '__main__':
    main_vid(
        #dir_image="/home/veily3/LIGAN/VeilyCV/test/test_517/lg_image",
        pth_video="/home/veily3/LIGAN/VeilyCV/test/test_517/liganmp4.mp4", 
        dir_output_project="/home/veily3/LIGAN/VeilyCV/test/test_517/LG_video", 
        dir_output_obj="/home/veily3/LIGAN/VeilyCV/flask/obj_result",
        name_output_obj="LG.obj",
        known_box_constraint="a4",
        unitized_length="cm"
        )

