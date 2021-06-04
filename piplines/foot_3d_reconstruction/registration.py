
from scipy.optimize import minimize
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
from src.piplines.foot_3d_reconstruction.debug import debug_vis, separator
from scipy.optimize import minimize
import open3d as o3d




@ separator
def calc_trajectory_by_rltv(dir_input, df_corners_regst, K, size_chessboard, size_img, M_rltv, name_color_std="color_01", len_chessboard=0.02, dist=np.zeros(5), debug=False, dir_debug=None):        
    if dir_debug is not None:
        os.makedirs(dir_debug) if not os.path.exists(dir_debug) else None
    else:
        pass

    df_pts3d = pd.DataFrame(index=df_corners_regst.index, columns=df_corners_regst.columns)

    grid_3d = np.mgrid[0:size_chessboard[0], 0:size_chessboard[1], 0:1] * len_chessboard
    grid_3d = grid_3d.T.reshape((-1, 3))

    names_colors = df_corners_regst.columns
    bgr_colors = EasyDict({names_colors[0]: (0, 0, 255), names_colors[1]: (255, 0, 0)})

    cam0 = Camera.PinholeCamera(size_img[0], size_img[1], K)
    name_img_init = (df_corners_regst.isna().sum(axis=1)==False).index.tolist()[0]

    df_traj = pd.DataFrame(index=df_corners_regst.index, columns=["traj", "error"])
    for name_img in df_corners_regst.index:
        img = cv2.imread(os.path.join(dir_input, name_img))
        img_out = img.copy()

        print("*"*32)
        ic(name_img)
        """
        Initialize
        """
        if name_img == name_img_init:

            df_corners_init = df_corners_regst.loc[name_img]
            if name_color_std  not in names_colors:
                cam0.set_extrinsic_matrix_identity()
                df_traj.loc[name_img, "traj" ] = np.NaN
                df_traj.loc[name_img, "error"] = np.NaN
                continue

            for name_color in names_colors:
                pts_init = np.asarray(df_corners_regst.loc[name_img, name_color])

                [_, rvec, tvec] = cv2.solvePnP(np.expand_dims(grid_3d, axis=0), pts_init, K, dist)

                cam0.extrinsic.set_rtvec(rvec, tvec)
                img_out = cam0.project_points_on_image(grid_3d, color=bgr_colors[name_color], img=img_out, radius=2)

                if name_color == name_color_std:
                    df_traj.loc[name_img, "traj" ] = cam0.extrinsic._mat_4x4()
                    df_traj.loc[name_img, "error"] = 0
                    if debug:
                        img_out = cam0.project_points_on_image(grid_3d, color=bgr_colors[name_color], img=img_out, radius=5)
                        img_out = cam0.project_axis_on_image(len_chessboard*size_chessboard[0], img=img_out)
                    else:
                        pass
                else:
                    pass
                
            ic("Initialize.")
            if debug:
                pth_out = os.path.join(dir_debug, name_img)
                cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
                cv2.imshow("debug", img_out)
                cv2.waitKey(debug)
                cv2.imwrite(pth_out, img_out)
                ic(pth_out)

        else:
            isfind_corner_in_colors = df_corners_regst.loc[name_img].isna().apply(lambda x: not x)
            ic(isfind_corner_in_colors)
            if not bool(isfind_corner_in_colors.sum()): # dont any any corners for each color
                df_traj.loc[name_img, "traj" ] = np.NaN
                df_traj.loc[name_img, "error"] = np.NaN
            else:
                names_color_within = isfind_corner_in_colors[isfind_corner_in_colors].index
                pts_std = np.asarray(df_corners_init.loc[names_color_within].values.tolist())
                df_pts_src = df_corners_regst.loc[name_img, names_color_within]


                if name_color_std in names_color_within:
                    cam1 = cam0.copy()

                    pts_src = np.asarray(df_pts_src[[name_color_std]].tolist())
                    [_, rvec, tvec] = cv2.solvePnP(np.expand_dims(grid_3d, axis=0), pts_src, cam1.intrinsic._mat_3x3(), dist)
                    cam1.extrinsic.set_rtvec(rvec, tvec)

                    df_traj.loc[name_img, "traj" ] = cam1.extrinsic._mat_4x4()
                    df_traj.loc[name_img, "error"] = np.average(np.linalg.norm(cam1.project_points(grid_3d) - pts_src, axis=1))

                    img_out = cam1.project_points_on_image(grid_3d, color=bgr_colors[name_color_std], img=img_out, radius=2)
                    img_out = cam1.project_axis_on_image(len_chessboard*size_chessboard[0], img=img_out, width_line=10)
                    ic(df_traj.loc[name_img, "error"])

                    debug_vis(debug=debug, dir_debug=dir_debug, name_img=name_img, img_out=img_out)
                    continue

                else: # estimate by rltv
                    cam2 = cam0.copy()

                    name_color = df_pts_src.index[~df_pts_src.index.isin([name_color_std])]
                    pts_src = np.asarray(df_pts_src[name_color].tolist())
                    [_, rvec, tvec] = cv2.solvePnP(np.expand_dims(grid_3d, axis=0), pts_src, cam2.intrinsic._mat_3x3(), dist)
                    cam2.extrinsic.set_rtvec(rvec, tvec)
                    M2 = cam2.extrinsic._mat_4x4() @ M_rltv  # M1 @ X
                    cam2.set_extrinsic_matrix(M2)

                    df_traj.loc[name_img, "traj" ] = cam2.extrinsic._mat_4x4()
                    df_traj.loc[name_img, "error"] = np.average(np.linalg.norm(cam2.project_points(grid_3d) - pts_src, axis=1))

                    img_out = cam2.project_points_on_image(grid_3d, color=bgr_colors[name_color_std], img=img_out, radius=2)
                    img_out = cam2.project_axis_on_image(len_chessboard*size_chessboard[0], img=img_out, width_line=10)
                    ic(df_traj.loc[name_img, "error"])
                   
                    debug_vis(debug=debug, dir_debug=dir_debug, name_img=name_img, img_out=img_out)
    return df_traj, df_pts3d


@ separator
def calc_trajectory_by_essential(dir_input, df_corners_regst, K, size_chessboard, size_img, name_color_std="color_01", len_chessboard=0.02, dist=np.zeros(5), debug=False, dir_debug=None): 
    df_pts3d = pd.DataFrame(index=df_corners_regst.index, columns=df_corners_regst.columns)

    grid_3d = np.mgrid[0:size_chessboard[0], 0:size_chessboard[1], 0:1] * len_chessboard
    grid_3d = grid_3d.T.reshape((-1, 3))

    names_colors = df_corners_regst.columns
    bgr_colors = EasyDict({names_colors[0]: (0, 0, 255), names_colors[1]: (255, 0, 0)})

    cam0 = Camera.PinholeCamera(size_img[0], size_img[1], K)
    name_img_init = (df_corners_regst.isna().sum(axis=1)==False).index.tolist()[0]

    df_traj = pd.DataFrame(index=df_corners_regst.index, columns=["traj", "error"])
    for name_img in df_corners_regst.index:
        img = cv2.imread(os.path.join(dir_input, name_img))
        img_out = img.copy()

        """
        Initialize
        """
        if name_img == name_img_init:

            df_corners_init = df_corners_regst.loc[name_img]
            if name_color_std  not in names_colors:
                cam0.set_extrinsic_matrix_identity()
                df_traj.loc[name_img, "traj" ] = cam0.extrinsic._mat_4x4()
                df_traj.loc[name_img, "error"] = 0
                continue

            for name_color in names_colors:
                pts_init = np.asarray(df_corners_regst.loc[name_img, name_color])

                [_, rvec, tvec] = cv2.solvePnP(np.expand_dims(grid_3d, axis=0), pts_init, K, dist)

                cam0.extrinsic.set_rtvec(rvec, tvec)
                img_out = cam0.project_points_on_image(grid_3d, color=bgr_colors[name_color], img=img_out, radius=2)

                if name_color == name_color_std:
                    df_traj.loc[name_img, "traj" ] = cam0.extrinsic._mat_4x4()
                    df_traj.loc[name_img, "error"] = 0
                    img_out = cam0.project_points_on_image(grid_3d, color=bgr_colors[name_color_std], img=img_out, radius=5)
                    img_out = cam0.project_axis_on_image(len_chessboard*size_chessboard[0], img=img_out, width_line=10)
                else:
                    pass
            debug_vis(debug=debug, dir_debug=dir_debug, name_img=name_img, img_out=img_out)
        else:
            isfind_corner_in_colors = df_corners_regst.loc[name_img].isna().apply(lambda x: not x)
            if not bool(isfind_corner_in_colors.sum()): # dont any any corners for each color
                df_traj.loc[name_img, "traj" ] = np.NaN
                df_traj.loc[name_img, "error"] = np.NaN
            else:
                names_color_within = isfind_corner_in_colors[isfind_corner_in_colors].index
                pts_std = np.asarray(df_corners_init.loc[names_color_within].values.tolist())
                pts_src = np.asarray(df_corners_regst.loc[name_img, names_color_within].values.tolist())
                [E, mask] = cv2.findEssentialMat(pts_std, pts_src, cameraMatrix=K, method=cv2.LMEDS)
                [ret, R, t, mask] = cv2.recoverPose(E, pts_std, pts_src, K) 
                """
                !!! t is a normalized vector !!!
                """

                cam1 = cam0.copy()
                cam1.extrinsic.set_R_tvec(R, t)
                cam2 = cam0.copy()
                M1 = cam1.extrinsic._mat_4x4() @ cam0.extrinsic._mat_4x4()
                cam1.set_extrinsic_matrix(M1)


                pts4d = cv2.triangulatePoints(cam0._projection_mat_3x4(), cam1._projection_mat_3x4(), pts_std, pts_src)
                pts4d = pts4d/pts4d[-1]
                pts3d = pts4d[:3].T

                df_pts3d.loc[name_img, names_color_within] = [pts3d]
 
                img_out = cam1.project_points_on_image(pts3d, color=(0, 0, 0), img=img_out, radius=2)
                img_out = cam1.project_points_on_image(grid_3d, color=(0, 0, 0), img=img_out, radius=5)
                img_out = cam1.project_axis_on_image(len_chessboard*size_chessboard[0], img=img_out, width_line=10)

                df_traj.loc[name_img, "traj" ] = cam1.extrinsic._mat_4x4()
                df_traj.loc[name_img, "error"] = np.average(np.linalg.norm(cam1.project_points(pts3d) - pts_src, axis=1))

                if name_color_std in names_color_within:
                    [_, rvec, tvec] = cv2.solvePnP(np.expand_dims(grid_3d, axis=0), pts_src, cam2.intrinsic._mat_3x3(), dist)
                    cam2.extrinsic.set_rtvec(rvec, tvec)
                    img_out = cam2.project_points_on_image(grid_3d, color=bgr_colors[name_color_std], img=img_out, radius=5)
                    img_out = cam2.project_axis_on_image(len_chessboard*size_chessboard[0], img=img_out, width_line=10)
                   

                print("*"*32)
                ic(name_img, names_color_within, df_traj.loc[name_img, "error"])
                debug_vis(debug=debug, dir_debug=dir_debug, name_img=name_img, img_out=img_out)
    return df_traj, df_pts3d


@ separator
def save_trajectory(dir_output, df_trajectory):
    pth_output_json = os.path.join(dir_output, "data", "trajectory.json")
    data = df_trajectory.to_json(orient="index")
    with open(pth_output_json, "w") as fp:
        json.dump(json.loads(data), fp, indent=4)
    ic(pth_output_json)

    return

 
@ separator
def load_trajectory(dir_input, ):
    pth_input_json = os.path.join(dir_input, "trajectory.json")
    trajectory = pd.read_json(pth_input_json).T
    ic(pth_input_json)
    return trajectory


@ separator
def save_points3d(dir_output, df_points3d):
    pth_output_json = os.path.join(dir_output, "data", "points3d.json")
    data = df_points3d.to_json(orient="index")
    with open(pth_output_json, "w") as fp:
        json.dump(json.loads(data), fp, indent=4)
    ic(pth_output_json)
    return



def main(dir_calib_root, dir_fit_root, size_chessboard, colors, sub_corner=False, name_color_std="color_01", debug=False):
    dir_raw = os.path.join(dir_fit_root, "raw")

    [K, Ms_rltv, size_img] = calibrate_dual.load_calib_dual(dir_calib_root)

    """
    segment images by color in lab space
    """
    dir_color = os.path.join(dir_fit_root, "colors")
    if len(glob.glob(os.path.join(dir_color, "*/*.jpg"))) < 1:
        dirs_color = calibrate_dual.segment_images(dir_raw=dir_raw, colors=colors, delta=np.array([[40, 15, 15], [50, 50, 25]]), seg_inv=True, debug=debug)
        ic(calibrate_dual.segment_images)
    else:
        dirs_color = glob.glob(os.path.join(dir_color, "*"))
        dirs_color.sort()
        ic(dirs_color)
    print("*"*64+"\n")

    """
    1. read corners from json file if it exists
    2. if json file not exists, find corners and save in json file
    """
    pth_corners_json_regst = os.path.join(dir_fit_root, "data", "corners.json")
    if not os.path.exists(pth_corners_json_regst):
        dir_debug = os.path.join(dir_fit_root, "debug", "corners")
        pth_corners_json_regst = calibrate_dual.find_corners_dual(dir_raw=dir_raw, dirs_color=dirs_color, size_chessboard=size_chessboard, sub_corner=sub_corner, debug=debug, dir_debug=dir_debug)
    else:
        ic(pth_corners_json_regst)
    print("*"*64+"\n")

    """
    registrate
    """
    pth_corners_json_calib = os.path.join(dir_calib_root, "data", "corners.json")
    ic(pth_corners_json_calib)
    df_pts2d_regst  = calibrate_dual.load_corners_dual(pth_corners_json_regst)

    dir_debug = os.path.join(dir_fit_root, "debug", "traj")
    [df_traj, df_pts3d] = calc_trajectory_by_rltv(dir_input=dir_raw, df_corners_regst=df_pts2d_regst, K=K, M_rltv=Ms_rltv[0], name_color_std=name_color_std, size_chessboard=size_chessboard, size_img=size_img, debug=debug, dir_debug=dir_debug)
    save_trajectory(dir_output=dir_fit_root, df_trajectory=df_traj)
    save_points3d(dir_output=dir_fit_root, df_points3d=df_pts3d)
    print("*"*64+"\n")
    return

if __name__ == '__main__':

    colors = np.array([by_colors.red_lab, by_colors.blue_lab])
    size_chessboard = (9, 6)
    dir_calib_root  = "/media/veily3/Data/ljj/foots_3d/dataset/calib"
    dir_input = "/media/veily3/Data/ljj/foots_3d/dataset/fit"
    size_image = (1440, 1080)#(2736, 3648) # (960, 544)
    main(dir_calib_root, dir_input, size_chessboard,  sub_corner=True, colors=colors, name_color_std="color_02", debug=0)