
import argparse
import sys
import time
from pathlib import Path

from enum import Enum, unique
import numpy as np
import open3d as o3d
from cv2 import aruco as aruco
from easydict import EasyDict
from icecream import ic

dir_dnn = Path("/home/veily3/LIGAN/detectron")
sys.path.append(str(dir_dnn))
import predict_foot_PR

sys.path.append("../..")
from tools import rename_images, video2images
from piplines.foot_3d_reconstruction import (calibrate_camera_aruco,
                                             reconstruction_aruco,
                                             registration_aruco)
from piplines.foot_3d_reconstruction.reconstruction_aruco import ARUCO_BOARD_A3, ARUCO_BOARD_A4

zc1 = np.array([-np.inf, 0])
zc2 = np.array([0.05, np.inf,])
zc3 = np.array([0, np.inf])


@ unique
class UnitizedLength(Enum):
    METER      = 1
    CENTIMETER = 100
    MILLIMETER = 1000
UNITIZED_LENGTH_M  = UnitizedLength.METER
UNITIZED_LENGTH_CM = UnitizedLength.CENTIMETER
UNITIZED_LENGTH_MM = UnitizedLength.MILLIMETER





def main(dir_output_project, dir_output_obj, name_output_obj, aruco_board_type=ARUCO_BOARD_A3, unitized_length=UNITIZED_LENGTH_M, **kwargs):
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
        pth_video = kwargs["pth_video"]

    dir_project = Path(dir_output_project)
    if   (pth_video is not None) and (dir_image is None):
        print("传入的是视频。")
        dir_output_raw = Path(dir_output_project, "raw")
        video2images.viedo2images(pth_video=pth_video, dir_output=dir_output_raw, downsample=20)        
        print("视频采样成功。")
    elif (dir_image is not None) and (pth_video is None):
        print("传入的是图片。")
        dir_output_raw = Path(dir_output_project, "raw")
        rename_images.rename_images(dir_image, dir_output_raw, shape=None, suffix=".png")
        print("图片上传成功。")
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
        p=0.3
    )


    """
    Reconstruction
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    if   aruco_board_type == ARUCO_BOARD_A3:      
        board_aruco = aruco.GridBoard_create(markersX=13, markersY=9, markerLength=0.026, markerSeparation=0.0045, dictionary=aruco_dict)
    elif aruco_board_type == ARUCO_BOARD_A4:
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
        aruco_board_type=aruco_board_type,
        prop=0.8
    )

    mesh.scale(unitized_length.value, center=mesh.get_center()) # cm? mm?
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
    o3d.io.write_triangle_mesh(str(pth_output), mesh, write_vertex_colors=True, write_vertex_normals=True, print_progress=True)
    ic(pth_output)
    return length_foot, width_foot




if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser(description="Args of 3d Reconstruction")
    parser.add_argument("--input_dir_image",  type=str, help="input_dir image")
    parser.add_argument("--input_pth_video",  type=str, help="input_pth video")
    parser.add_argument("--output_dir_project", type=str, help="output_dir_project")
    parser.add_argument("--aruco_board_type", type=str, default="a3", help="aruco_board_type")
    parser.add_argument("--dir_output_obj", type=str, default="/home/veily3/LIGAN/VeilyCV/flask/obj_result", help="dir_output_obj")
    
    args = parser.parse_args()

    args.input_dir_image    = "/home/veily3/LIGAN/VeilyCV/test/test_524/wechat_1045"
    #args.input_pth_video   = "/home/veily3/LIGAN/VeilyCV/test/test_517/v3.mp4"
    args.output_dir_project = "/home/veily3/LIGAN/VeilyCV/test/test_524/WECHAT_1045"
    name_output_obj         = Path(args.output_dir_project).name + ".obj"
    args.aruco_board_type   = "a4"


    if (args.input_dir_image is not None) and (args.input_pth_video is not None):
        raise ValueError("use input_dir_image or input_pth_video")
    aruco_board_type = ARUCO_BOARD_A3 if args.aruco_board_type == "a3" else ARUCO_BOARD_A4


    main(
        dir_image=args.input_dir_image,
        pth_video=args.input_pth_video, 
        dir_output_project=args.output_dir_project, 
        dir_output_obj=args.dir_output_obj,
        name_output_obj=name_output_obj,
        aruco_board_type=aruco_board_type,
        unitized_length=UNITIZED_LENGTH_M,
    )
