
from src import Camera, geometries
import numpy as np
import cv2
import os
import glob
import json
import pandas as pd
from easydict import EasyDict
from icecream import ic
from src.segment import by_colors
from src.piplines.foot_3d_reconstruction import calibrate_dual



def calc_trajectory(dir_input, subconers, K, M_rltv, size_chessboard, size_img, len_chessboard=0.02, dist=None, debug=False):        
    [H, W] = size_img
    cam = Camera.PinholeCamera(H, W, K)
    grid_3d = np.mgrid[0:size_chessboard[0], 0:size_chessboard[1], 0:1] * len_chessboard
    grid_3d = grid_3d.T.reshape((-1, 3))

    traj = EasyDict({})
    for name_img in subconers.keys():
        [ret1, ret2] = [False, False]
        traj[name_img] = EasyDict({})

        # load rtvecs
        for name_color in subconers[name_img].keys():
            traj[name_img][name_color] = EasyDict({})
            pts = np.array(subconers[name_img][name_color])
            if np.isnan(pts).any():
                rvec = np.full((3), np.nan)
                tvec = np.full((3), np.nan)
            else:
                ret, rvec, tvec =  cv2.solvePnP(grid_3d, pts, K, np.zeros(5), cv2.SOLVEPNP_EPNP)
                rvec = rvec.reshape(-1)
                tvec = tvec.reshape(-1)
            traj[name_img][name_color].rvec = rvec.tolist()
            traj[name_img][name_color].tvec = tvec.tolist()

        # set rtvec relatively
        is_nan1 = np.isnan(traj[name_img]["color_01"].rvec).any()
        is_nan2 = np.isnan(traj[name_img]["color_02"].rvec).any()
        if is_nan1 ^ is_nan2: # XOR
            if is_nan1:
                rvec2 = np.array(traj[name_img]["color_02"].rvec)
                tvec2 = np.array(traj[name_img]["color_02"].tvec)
                R2 = geometries.r_to_R(rvec2)
                T2 = geometries.t_to_T(tvec2)
                M_2to1 = M_rltv[1]
                M1 = T2 @ R2 @ M_2to1
                [rvec1, tvec1] = geometries.decompose_transform_matrix_to_rtvec(M1)
                traj[name_img]["color_01"].rvec = rvec1.tolist()
                traj[name_img]["color_01"].tvec = tvec1.tolist()

            elif is_nan2:
                rvec1 = np.array(traj[name_img]["color_01"].rvec)
                tvec1 = np.array(traj[name_img]["color_01"].tvec)
                R1 = geometries.r_to_R(rvec1)
                T1 = geometries.t_to_T(tvec1)
                M_1to2 = M_rltv[0]
                M2 = T1 @ R1 @ M_1to2
                [rvec2, tvec2] = geometries.decompose_transform_matrix_to_rtvec(M2)
                traj[name_img]["color_02"].rvec = rvec2.tolist()
                traj[name_img]["color_02"].tvec = tvec2.tolist()

    [dir_parent, _] = os.path.split(dir_input)
    for name_img in subconers.keys():
        pth_img = os.path.join(dir_parent, "show", name_img)
        img = cv2.imread(pth_img)
        img_out = img.copy()
        for name_color in traj[name_img].keys():
            rvec = np.array(traj[name_img][name_color].rvec)
            tvec = np.array(traj[name_img][name_color].tvec)

            if np.isnan(rvec).any():
                break

            cam.set_rtvec(rvec, tvec)
            if   name_color == "color_01":
                coners = np.array(subconers[name_img][name_color]).astype(np.float32)
                cv2.drawChessboardCorners(image=img_out, patternSize=size_chessboard, corners=coners, patternWasFound=False)
                img_out = cam.project_points_on_image(points3d=grid_3d, color=(255, 0, 0), img=img_out, radius=2, thickness=-1)
            elif name_color == "color_02":
                coners = np.array(subconers[name_img][name_color]).astype(np.float32)
                cv2.drawChessboardCorners(image=img_out, patternSize=size_chessboard, corners=coners, patternWasFound=False)
                img_out = cam.project_points_on_image(points3d=grid_3d, color=(0, 255, 0), img=img_out, radius=2, thickness=-1)
            img_out = cam.project_axis_on_image(img=img_out, unit_length=5*0.029, width_line=1)
            cv2.imwrite(pth_img, img_out)
            
        if 1:
            cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
            cv2.imshow("debug", img_out)
            cv2.waitKey(debug)
        else:
            continue

    return traj


def save_trajectory(dir_output, trajectory):
    pth_output_json = os.path.join(dir_output, "trajectory.json")
    with open(pth_output_json, "w") as fp:
        json.dump(trajectory, fp, indent=4)
        ic(pth_output_json)
    return


def load_trajectory(dir_input, ):
    pth_input_json = os.path.join(dir_input, "trajectory.json")
    trajectory = pd.from_json()    
    return trajectory


def main(dir_calib_root, dir_input, size_chessboard, colors, sub_corner=False, debug=False):
    dir_raw = os.path.join(dir_input, "raw")

    [K, M_rltv, size_img] = calibrate_dual.load_calib_dual(dir_calib_root)

    dir_color = os.path.join(dir_input, "colors")
    if len(glob.glob(os.path.join(dir_color, "*/*.jpg"))) < 1:
        dirs_color = calibrate_dual.segment_images(dir_raw=dir_raw, colors=colors, delta=np.array([[40, 15, 15], [50, 50, 25]]), seg_inv=True, debug=debug)
        ic(calibrate_dual.segment_images)
    else:
        dirs_color = glob.glob(os.path.join(dir_color, "*"))
        dirs_color.sort()
        ic(dirs_color)

    pth_subconers_json = os.path.join(dir_input, "coners.json")
    if not os.path.exists(pth_subconers_json):
        pth_subconers_json = calibrate_dual.find_coners_dual(dir_raw, dirs_color, size_chessboard, sub_corner=sub_corner, debug=debug)
    else:
        ic(pth_subconers_json)

    subconers = calibrate_dual.load_corners_dual(pth_subconers_json)
    traj = calc_trajectory(dir_input=dir_raw, subconers=subconers, K=K, M_rltv=M_rltv, size_chessboard=size_chessboard, size_img=size_img, debug=debug)
    save_trajectory(dir_output=dir_input, trajectory=traj)
    return

if __name__ == '__main__':

    colors = np.array([by_colors.red_lab, by_colors.blue_lab])
    size_chessboard = (9, 6)
    dir_calib_root  = "/home/veily3/LIGAN/VeilyCV/test/test_507/calib_subpixel"
    dir_input = "/home/veily3/LIGAN/VeilyCV/test/test_507/fit/test1_subpixel"
    size_image = (1440, 1080)#(2736, 3648) # (960, 544)
    main(dir_calib_root, dir_input, size_chessboard,  sub_corner=True, colors=colors, debug=0)