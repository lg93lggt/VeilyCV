
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from cv2 import aruco as aruco
from icecream import ic

sys.path.append("..")
import calibrate_camera_aruco
import reconstruction_aruco
import registration_aruco

dir_dnn = Path("/home/veily3/LIGAN/detectron")
sys.path.append(str(dir_dnn))
import predict_foot_PR


def main(board_aruco, dir_input, debug=False):

    zc1 = np.array([-np.inf, 0])
    zc2 = np.array([0.05, np.inf,])
    zc3 = np.array([0, np.inf])
    
    calibrate_camera_aruco.main(
        board_aruco=board_aruco,
        dir_input=dir_input, 
        debug=debug,
    )
    reconstruction_aruco.main(
        dir_fit_root=dir_input, 
        dir_calib_root=dir_input, 
        debug=debug, 
        size_img=None, 
        prop=1,
        z_constraint=zc3,
        estimate_box=False
    )
    return


if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser(description="Input Dir")
    parser.add_argument("--dir", type=str, help="Input Dir")
    args = parser.parse_args()

    dir_input = args.dir if args.dir is not None else "/home/veily3/LIGAN/VeilyCV/test/test_509"

    dnn_yaml_path       = Path(dir_dnn, "detectron/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
    dnn_model_path      = Path(dir_dnn, "output/footmask_PR", "1_model_final.pth")
    dnn_pthes_input_img = sorted(Path(dir_input, "raw").glob('*.*'))
    dnn_output_path     = Path(dir_input,'mask')
    predict_foot_PR.main(
        yaml_path=str(dnn_yaml_path), 
        model_path=str(dnn_model_path), 
        pthes_input_img=dnn_pthes_input_img, 
        output_path=str(dnn_output_path)
    )
    t1 = time.time()
    ic(t1 - t0)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    board_aruco = aruco.GridBoard_create(markersX=13, markersY=9, markerLength=0.026, markerSeparation=0.0045, dictionary=aruco_dict)

    main(board_aruco=board_aruco, dir_input=dir_input, debug=0)
    t2 = time.time()
    ic(t2 - t1)

    total_time = t2 - t0
    ic(total_time)
