
import numpy as np
import cv2
import os
import glob
import json
import pandas as pd
from easydict import EasyDict
from icecream import ic
from src.segment import by_colors
from scipy.optimize import minimize
from src import Camera, geometries
from src.utils.debug import debug_separator, debug_vis


def cvt_df_values_to_np(df):
    array = np.asarray(df.tolist())
    return array

def del_nan_points(points2d):
    points2d[~np.isnan(points2d).any(axis=1)[:, 0]]
    return points2d


def objective_func(params, kwargs):

    K = np.eye(4)
    pts2d_obj = kwargs["pts2d_obj"]
    pts3d_homo = kwargs["pts3d_homo"]
    K[:3, :3] = kwargs["K"]
    rvecs = kwargs["rvecs"]
    tvecs = kwargs["tvecs"]
    Ms = geometries.rtvecs_to_transform_matrices(rvecs, tvecs)

    #params = EasyDict(params)
    R = geometries.rvec_to_Rmat(params[0])
    T = geometries.t_to_T(params[1])

    errors = np.ones(Ms.shape[0])
    for idx, M in enumerate(Ms):
        pts2d_dst = K @ M @ T @ R @ pts3d_homo
        pts2d_dst = pts2d_dst / pts2d_dst[2]
        pts2d_dst = pts2d_dst[:2].T
        errors[idx] = np.average(np.linalg.norm(
            pts2d_obj[idx] - pts2d_dst, axis=1))
    # ic(np.average(errors))
    return np.average(errors)


def objective_func_with_zero_z_constraint(params, kwargs):

    K = np.eye(4)
    pts2d_obj = kwargs["pts2d_obj"]
    pts3d_homo = kwargs["pts3d_homo"]
    K[:3, :3] = kwargs["K"]
    rvecs = kwargs["rvecs"]
    tvecs = kwargs["tvecs"]
    Ms = geometries.rtvecs_to_transform_matrices(rvecs, tvecs)

    #params = EasyDict(params)
    R = geometries.Rz(params[2], in_degree=True)
    T = geometries.t_to_T(np.array([params[0], params[1], 0]))

    errors = np.ones(Ms.shape[0])
    for idx, M in enumerate(Ms):
        pts2d_dst = K @ M @ T @ R @ pts3d_homo
        pts2d_dst = pts2d_dst / pts2d_dst[2]
        pts2d_dst = pts2d_dst[:2].T
        errors[idx] = np.average(np.linalg.norm(
            pts2d_obj[idx] - pts2d_dst, axis=1))
    # ic(np.average(errors))
    return np.average(errors)


def objective_func_Transform(params, kwargs):

    rvecs = kwargs["rvecs"]
    tvecs = kwargs["tvecs"]
    
    Ms1 = geometries.rtvecs_to_transform_matrices(rvecs.color_01.reshape((-1, 3)), tvecs.color_01.reshape((-1, 3)))
    Ms2 = geometries.rtvecs_to_transform_matrices(rvecs.color_02.reshape((-1, 3)), tvecs.color_02.reshape((-1, 3)))
    

    R = geometries.rvec_to_Rmat(params[:3])
    T = geometries.t_to_T(params[3:])

    X = T@R
    _Ms2 = Ms1 @ X
    [_rs, _ts] = geometries.decompose_transform_matrices_to_rtvecs(_Ms2)
    _Rs = geometries.rs_to_Rs(_rs, shape=(3, 3))
    error_t = np.average(np.linalg.norm(_ts - tvecs.color_02.reshape((-1, 3)), axis=1))
    error_R =  geometries.distance_list_SO3(_Rs, Ms2[:, :3, :3])

    error = error_R + error_t
    return error

@ debug_separator
def segment_images(dir_raw, colors, seg_inv=False, delta=50, suffix=".jpg", debug=False):

    [dir_parent, _] = os.path.split(dir_raw)

    dirs_color = []
    for [idx_color, _] in enumerate(colors):
        dir_color = os.path.join(dir_parent, "colors",
                                 "{:0>2d}".format(idx_color+1))
        if not os.path.exists(dir_color):
            os.makedirs(dir_color)
        dirs_color.append(dir_color)

    pthes_img = glob.glob(os.path.join(dir_raw, "*"+suffix))
    pthes_img.sort()

    for pth_img in pthes_img:
        [_, name_img] = os.path.split(pth_img)
        img = cv2.imread(pth_img)
        if seg_inv:
            imgs_roi = by_colors.segment_by_colors_inv(img, colors, delta=delta, debug=debug)
        else:
            imgs_roi = by_colors.segment_by_colors(img, colors, delta=delta, debug=debug)

        for [idx_color, _] in enumerate(colors):
            img_roi = imgs_roi[idx_color]
            img_roi = np.where(img_roi == [0, 0, 0], 255, img_roi)
            pth_output = os.path.join(dirs_color[idx_color], name_img)
            ret = cv2.imwrite(pth_output, img_roi)
            ic(pth_output, ret)

    return dirs_color


@ debug_separator
def find_corners_dual(dir_raw, dirs_color, size_chessboard, suffix=".jpg", sub_corner=False, debug=False, dir_debug=None):
    [dir_parent, _] = os.path.split(dir_raw)
    pthes_img = glob.glob(os.path.join(dir_raw, "*"+suffix))
    pthes_img.sort()

    colors = ["color_" + os.path.split(x)[1] for x in dirs_color]

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-2)

    pts = EasyDict({})
    for [_, pth_img] in enumerate(pthes_img):
        [_, name_img] = os.path.split(pth_img)

        if debug:
            img = cv2.imread(pth_img)
            img_out = img.copy()
        else:
            pass

        pts[name_img] = EasyDict({})
        for idx_color in range(len(dirs_color)):
            pth_color = os.path.join(dirs_color[idx_color], name_img)
            roi_color = cv2.imread(pth_color)

            if roi_color is not None:
                ret_find_corners, corners = cv2.findChessboardCorners(
                    roi_color, size_chessboard, cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret_find_corners:
                    if sub_corner:
                        corners = cv2.cornerSubPix(cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY), corners, (5, 5), (-1, -1), criteria)
                    else:
                        corners = corners.copy()
                    corners = corners.reshape((-1, 2))

                    pts[name_img]["color_{:0>2d}".format(idx_color+1)] = corners.tolist()

                    if debug:
                        img_out = cv2.drawChessboardCorners(
                            image=img_out, patternSize=size_chessboard, corners=corners, patternWasFound=ret_find_corners)
                    else:
                        img_out = None
                else:
                    pts[name_img]["color_{:0>2d}".format(idx_color+1)] = np.NaN
                ic(name_img, idx_color, ret_find_corners)
            else:
                pass
            
        debug_vis(debug=debug, dir_debug=dir_debug, name_img=name_img, img_out=img_out)

    pth_output_json = os.path.join(dir_parent, "data", "corners.json")
    with open(pth_output_json, "w") as fp:
        json.dump(pts, fp, indent=4)
        ic(pth_output_json)
    return pth_output_json


@ debug_separator
def calibrate_dual(df_pts2d, size_chessboard, size_image, len_chessboard=0.02):
    grid_3d = np.mgrid[0:size_chessboard[0], 0:size_chessboard[1], 0:1] * len_chessboard
    grid_3d = grid_3d.T.reshape((-1, 3))

    [H, W] = size_image

    pts2d_all = []
    for name_color in df_pts2d.columns:
        pts2d_all += df_pts2d[name_color].tolist()
    pts2d_all = np.asarray(pts2d_all)
    n_imgs = pts2d_all.shape[0] // 2

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(np.asarray([grid_3d]*n_imgs*2).astype(np.float32), pts2d_all.astype(np.float32), (W, H), cameraMatrix=None, distCoeffs=np.zeros(5),
                                                     flags=cv2.CALIB_ZERO_TANGENT_DIST+cv2.CALIB_FIX_K1+cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3)
    ic(ret, K, dist)

    dist_rvecs = EasyDict({})
    dist_rvecs.color_01 = np.asarray(rvecs[:n_imgs])
    dist_rvecs.color_02 = np.asarray(rvecs[n_imgs:])

    dist_tvecs = EasyDict({})
    dist_tvecs.color_01 = np.asarray(tvecs[:n_imgs])
    dist_tvecs.color_02 = np.asarray(tvecs[n_imgs:])

    zero_z = 0
    x0_z0    = {"x": 0, "y": 0, "rz": 0}
    x0_rtvec = EasyDict({"rvec": np.zeros(3), "tvec": np.zeros(3)})
    pts3d_homo = geometries.to_homo(P=grid_3d)
    if zero_z:
        func = objective_func_with_zero_z_constraint
        x0 = x0_z0
    else:
        Ms1 = geometries.rtvecs_to_transform_matrices( dist_rvecs.color_01,  dist_tvecs.color_01)
        Ms2 = geometries.rtvecs_to_transform_matrices( dist_rvecs.color_02,  dist_tvecs.color_02)
        Xs0 = Ms2 @ np.linalg.inv(Ms1)
        X0 = geometries.average_SE3(Xs0)
        func = objective_func_Transform
        x0 = x0_rtvec
        x0.rvec, x0.tvec = geometries.decompose_transform_matrix_to_rtvec(X0)
    cvt_df_values_to_np(df_pts2d["color_01"])
    res12 = minimize(func, x0=list(x0.values()), args={
                   "K": K, "rvecs": dist_rvecs, "tvecs": dist_tvecs, "pts3d_homo": pts3d_homo, "pts2d_obj": cvt_df_values_to_np(df_pts2d["color_02"])})
    # res21 = minimize(func, x0=list(x0.values()), args={
    #                "K": K, "rvecs": dist_rvecs.color_02, "tvecs": dist_tvecs.color_02, "pts3d_homo": pts3d_homo, "pts2d_obj": cvt_df_values_to_np(df_pts2d["color_01"])})
    
    if zero_z:
        R12 = geometries.Rz(res12.x[2], in_degree=True)
        T12 = geometries.t_to_T(np.array([res12.x[0], res12.x[1], 0]))

        Ms_rltv = [T12@R12, np.eye(4)]
        #Ms_rltv = [R12@T12, R21@T21]
    else:
        Ms_rltv = [ 
            geometries.rtvec_to_transform_matrix(res12.x[:3], res12.x[3:]),
             np.eye(4),
            ]
    ic(res12.fun, res12.x)
    return [K, dist_rvecs, dist_tvecs, Ms_rltv, dist]


@ debug_separator
def load_corners_dual(pth_corners):
    df_corners = pd.read_json(pth_corners).T
    df_corners.fillna(value=np.NaN, inplace=True)
    return df_corners  # (n_imgs*n_colors, w*h chessboard, 2)


@ debug_separator
def save_calib_results(dir_calib_root, K, dict_rvecs, dict_tvecs, Ms_rltv, dist, size_img):
    """
    Save calibration data to: "<dir_calib_root>/data/camera_params.json"
    """
    rvecs = dict_rvecs.copy()
    tvecs = dict_tvecs.copy()
    for key in rvecs.keys():
        rvecs[key] = rvecs[key].tolist()
    for key in tvecs.keys():
        tvecs[key] = tvecs[key].tolist()

    data = EasyDict({})
    data.intrinsic = K.tolist()
    data.rvecs = rvecs
    data.tvecs = tvecs
    data.dist = dist.tolist()
    data.height = size_img[0]
    data.width = size_img[1]
    data.color1_to_color2 = Ms_rltv[0].tolist()
    data.color2_to_color1 = Ms_rltv[1].tolist()

    pth_output = os.path.join(dir_calib_root, "data", "camera_params.json")
    with open(pth_output, "w") as fp:
        json.dump(data, fp, indent=4)
    ic(pth_output)
    return


@ debug_separator
def load_calib_dual(dir_calib_root):
    """
    Load calibration data from: "<dir_calib_root>/data/camera_params.json"
    """
    pth_camera_params = os.path.join(dir_calib_root, "data", "camera_params.json")
    Ms_rltv = []
    with open(pth_camera_params, "r") as fp:
        camera_params = EasyDict(json.load(fp))
        K = np.array(camera_params.intrinsic)
        Ms_rltv.append(np.array(camera_params.color1_to_color2))
        Ms_rltv.append(np.array(camera_params.color2_to_color1))
        size_img = [camera_params.height, camera_params.width]
    return [K, Ms_rltv, size_img]


@ debug_separator
def evaluate(dir_calib_root, K, dict_rvecs, dict_tvecs, Ms_rltv, len_chessboard, size_img, size_chessboard, dist=np.zeros(5), df_pts2d=None, debug=True, dir_debug=None):
    dir_raw = os.path.join(dir_calib_root, "raw",)
    pthes_raw = glob.glob(os.path.join(dir_raw, "*.jpg",))
    pthes_raw.sort()

    cam1 = Camera.PinholeCamera(size_img[0], size_img[1], K)
    cam2 = Camera.PinholeCamera(size_img[0], size_img[1], K)

    grid_3d = np.mgrid[0:size_chessboard[0], 0:size_chessboard[1], 0:1] * len_chessboard
    grid_3d = grid_3d.T.reshape((-1, 3))

    error = EasyDict({})
    error.color_01 = []
    error.color_02 = []
    

    for [idx, pth_raw] in enumerate(pthes_raw):
        img = cv2.imread(pth_raw)
        img_out = img.copy()

        rvec1 = dict_rvecs.color_01[idx]
        tvec1 = dict_tvecs.color_01[idx]
        rvec2 = dict_rvecs.color_02[idx]
        tvec2 = dict_tvecs.color_02[idx]
        M1 = geometries.rtvec_to_transform_matrix(rvec1, tvec1)
        M2 = geometries.rtvec_to_transform_matrix(rvec2, tvec2)

        M_2to1 = Ms_rltv[1]
        M1_ = M2 @ M_2to1

        # Orignal M1
        cam1.set_extrinsic_matrix(M1)
        img_out = cam1.project_points_on_image(grid_3d, (0, 0, 255), img=img_out, radius=2)


        # Orignal M2
        cam2.set_extrinsic_matrix(M2)
        img_out = cam2.project_points_on_image(grid_3d, (255, 0, 0), img=img_out, radius=2)

        # M2_ Transformed by M1
        M_1to2 = Ms_rltv[0]
        M2_ = M1 @ M_1to2
        cam2.set_extrinsic_matrix(M2_)
        img_out = cam2.project_points_on_image(grid_3d, (255, 0, 0), img=img_out, radius=5)
        img_out = cam2.project_axis_on_image(img=img_out, unit_length=(size_chessboard[0]-1)*len_chessboard, width_line=1)

        pts2d_dst = cam2.project_points(grid_3d)
        pts2d_dst = pts2d_dst.reshape((-1, 2))
        error.color_01.append(np.average(np.linalg.norm(pts2d_dst - np.asarray(df_pts2d.iloc[idx]["color_02"]), axis=1)))

        ic(idx, error.color_01[idx])
        if debug:
            while True:
                img_out = cam2.project_axis_on_image(img=img_out.copy(), unit_length=(size_chessboard[0]-1)*len_chessboard, width_line=3)
                cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
                cv2.imshow("debug", img_out)
                key = cv2.waitKey(100)
                if key == ord("a"):
                    cam2.set_extrinsic_matrix(cam2.extrinsic._mat_4x4()@geometries.t_to_T(-np.array([0.01, 0, 0])))
                if key == ord("s"):
                    cam2.set_extrinsic_matrix(cam2.extrinsic._mat_4x4()@geometries.t_to_T(-np.array([0, 0.01, 0])))
                if key == ord("d"):
                    cam2.set_extrinsic_matrix(cam2.extrinsic._mat_4x4()@geometries.Rz(-1, True))
                if key == ord("z"):
                    cam2.set_extrinsic_matrix(cam2.extrinsic._mat_4x4()@geometries.t_to_T(np.array([0.01, 0, 0])))
                if key == ord("x"):
                    cam2.set_extrinsic_matrix(cam2.extrinsic._mat_4x4()@geometries.t_to_T(np.array([0, 0.01, 0])))
                if key == ord("c"):
                    cam2.set_extrinsic_matrix(cam2.extrinsic._mat_4x4()@geometries.Rz(1, True))
                if key == ord("q"):
                    break
        [_, namea_img] = os.path.split(pth_raw)
        debug_vis(debug=debug, dir_debug=dir_debug, name_img=namea_img, img_out=img_out)
    ic(np.average(np.asarray(error.color_01)))
    return


def main(dir_calib_root, size_chessboard, size_image, colors, len_chessboard, sub_corner=False, debug=False):
    print("*"*64)

    """
    segment images by color in lab space
    """
    dir_calib_raw = os.path.join(dir_calib_root, "raw")
    dir_color = os.path.join(dir_calib_root, "colors")
    if len(glob.glob(os.path.join(dir_color, "*/*.jpg"))) < 1:
        dirs_color = segment_images(dir_raw=dir_calib_raw, colors=colors, delta=np.array([[40, 15, 15], [40, 40, 15]]), seg_inv=True, debug=debug)
        ic(segment_images)
    else:
        dirs_color = glob.glob(os.path.join(dir_color, "*"))
        dirs_color.sort()
        ic(dirs_color)
    print("*"*64)


    """
    1. read corners from json file if it exists
    2. if json file not exists, find corners and save in json file
    """
    pth_corners_json = os.path.join(dir_calib_root, "corners.json")
    if not os.path.exists(pth_corners_json):
        dir_debug = os.path.join(dir_calib_root, "debug", "find_corners")
        pth_corners_json = find_corners_dual(dir_calib_raw, dirs_color, size_chessboard, sub_corner=sub_corner, debug=debug, dir_debug=dir_debug)
        ic(find_corners_dual)
    else:
        ic(pth_corners_json)
    df_pts2d = load_corners_dual(pth_corners_json)
    print("*"*64)

    ic(calibrate_dual)
    [K, dict_rvecs, dict_tvecs, Ms_rltv, dist] = calibrate_dual(
        df_pts2d=df_pts2d, size_chessboard=size_chessboard, len_chessboard=len_chessboard, size_image=size_image)
    save_calib_results(dir_calib_root, K, dict_rvecs, dict_tvecs,
                       Ms_rltv=Ms_rltv, dist=dist, size_img=size_image)
    print("*"*64)

    ic(evaluate)
    dir_debug = os.path.join(dir_calib_root, "debug", "calibrate")
    evaluate(
        dir_calib_root, 
        K, dict_rvecs, dict_tvecs, dist=dist, Ms_rltv=Ms_rltv, 
        size_chessboard=size_chessboard, len_chessboard=len_chessboard, 
        df_pts2d=df_pts2d, 
        size_img=size_image, 
        debug=debug, dir_debug=dir_debug
    )
    print("*"*64)

if __name__ == '__main__':
    colors = np.array([by_colors.red_lab, by_colors.blue_lab])
    size_chessboard = (9, 6)
    dir_calib_root = "/media/veily3/Data/ljj/foots_3d/dataset/calib"
    size_image = (1440, 1080)#(2736, 3648) # (960, 544)
    main(dir_calib_root, size_chessboard, size_image, len_chessboard=0.02, colors=colors, sub_corner=True, debug=0)
