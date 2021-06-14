
import sys
from pathlib import Path
from typing import Iterable, List, Union

import cv2
import numpy as np
import pandas as pd
from cv2 import aruco
from icecream import ic
from matplotlib import pyplot as plt

sys.path.append("../..")
from src.Camera import PinholeCamera
from src.utils.debug import debug_separator, debug_vis


def without_nan(df: pd.DataFrame):
    df_without = df.loc[~df.isna()]
    return df_without

@ debug_separator
def find_corners_aruco(dictionary_aruco, dir_input, suffix_input=".*", debug=False, dir_debug=None):
    arucoParams = aruco.DetectorParameters_create()

    dir_input = Path(dir_input)
    dir_debug = Path(dir_debug, "corners")
    pthes_img = sorted(dir_input.glob("*" + suffix_input))
    names_img = sorted(x.name for x in dir_input.glob("*" + suffix_input))

    df_corners = pd.DataFrame(index=names_img, columns=[
                              "corners", "ids", "counter"])

    ic(dir_input)
    ic(dir_debug)
    for pth_input in pthes_img:
        img = cv2.imread(str(pth_input))
        [corners, ids, _] = aruco.detectMarkers(img, dictionary_aruco, parameters=arucoParams)
        n_ids = len(ids) if ids is not None else None

        df_corners.loc[pth_input.name, "corners"] = np.asarray(corners) if ids is not None else np.nan
        df_corners.loc[pth_input.name, "ids"    ] = np.asarray(ids)     if ids is not None else np.nan
        df_corners.loc[pth_input.name, "counter"] = n_ids               if ids is not None else np.nan

        print()
        ic(pth_input.name, n_ids)

        img_out = img.copy()
        aruco.drawDetectedMarkers(image=img_out, corners=corners, ids=ids)
        debug_vis(debug=debug, dir_debug=dir_debug,
                  name_img=pth_input.name, img_out=img_out)
    ic(df_corners.info())
    return df_corners


def save_corners(df_input, dir_output, prefix="corners", suffix=".pkl"):
    dir_output.mkdir(parents=True, exist_ok=True)
    pth_output = Path(dir_output, prefix + suffix)
    ic(pth_output)
    if suffix == ".json":
        df_input.to_json(pth_output, indent=4)
    elif suffix == ".xlsx":
        df_input.to_excel(pth_output)
    elif suffix == ".pkl":
        df_input.to_pickle(pth_output)
    return pth_output


@ debug_separator
def save_camera_parameters(df_input, dir_output, prefix="camera_params", suffix=".pkl"):
    return save_corners(df_input=df_input, dir_output=dir_output, prefix=prefix, suffix=suffix)


@ debug_separator
def save_trajectory(df_input, dir_output, prefix="trajectory", suffix=".pkl"):
    return save_corners(df_input=df_input, dir_output=dir_output, prefix=prefix, suffix=suffix)


def load_data(pth_input):
    pth_input = Path(pth_input)
    if pth_input.suffix == ".json":
        df_load = pd.read_json(pth_input, numpy=True).applymap(np.asarray)

    elif pth_input.suffix == ".xlsx":
        df_load = pd.read_excel(pth_input)

    elif pth_input.suffix == ".pkl":
        df_load = pd.read_pickle(pth_input)

    else:
        df_load = pd.DataFrame()
    ic(pth_input)
    return df_load


@ debug_separator
def load_corners(pth_input):
    df_coners = load_data(pth_input)
    ic(df_coners.info())
    return df_coners


@ debug_separator
def load_camera_parameters(pth_input):
    sr_camera_params = load_data(pth_input)
    ic(sr_camera_params)
    return sr_camera_params


@ debug_separator
def load_trajectory(pth_input):
    df_trajectory = load_data(pth_input)
    ic(df_trajectory.info())
    return df_trajectory


@ debug_separator
def calibrate_aruco(board_aruco, size_img, df_corners):

    corners = np.concatenate(without_nan(df_corners.loc[:, "corners"]).values, axis=0).astype(np.float32)
    ids     = np.concatenate(without_nan(df_corners.loc[:, "ids"]).values, axis=0)
    counter = without_nan(df_corners.loc[:, "counter"]).values.astype(int)
    print("data: [corners, ids, counter]")
    for data in [corners, ids, counter]:
        ic(data.shape, data.dtype)

    [error_rms, K, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, errors] = aruco.calibrateCameraArucoExtended(
        corners=corners,
        ids=ids,
        counter=counter,
        board=board_aruco,
        imageSize=(size_img[1], size_img[0]),
        cameraMatrix=None, distCoeffs=None,
        flags=cv2.CALIB_ZERO_TANGENT_DIST
    )
    ic(errors.shape, np.max(errors), np.argmax(errors), df_corners.index[np.argmax(errors)])
    # root mean square error
    ic(error_rms)
    
    sr_camera_params = pd.Series(index=["intrinsic", "distortion", "height", "width", "error", ])
    sr_camera_params.loc["intrinsic" ] = K.astype(np.object)
    sr_camera_params.loc["distortion"] = dist.astype(np.object)
    sr_camera_params.loc["height"    ] = np.array([size_img[0]], dtype=np.object)
    sr_camera_params.loc["width"     ] = np.array([size_img[1]], dtype=np.object)
    sr_camera_params.loc["error"     ] = stdDeviationsIntrinsics.astype(np.object)
    
    df_trajectory = pd.DataFrame(index=df_corners.index, columns=["rvec", "tvec", "error"])
    index_valid = ~df_corners.loc[:, "ids"].isna()
    df_trajectory.loc[index_valid, "rvec"]      = [r.astype(np.object) for r in rvecs]
    df_trajectory.loc[index_valid, "tvec"]      = [t.astype(np.object) for t in tvecs]
    df_trajectory.loc[index_valid, "error"]     = errors.flatten()

    ic(K, dist, error_rms)
    return [sr_camera_params, df_trajectory]


@ debug_separator
def evaluatle(board_aruco, dir_input, sr_camera_params: pd.Series, df_trajectory: pd.DataFrame, df_corners: pd.DataFrame, debug: int=True, dir_debug: Path or str=None):
    """evaluatle [summary]
    
    [extended_summary]
    
    Parameters
    ----------
    - `board_aruco` : [type]
        [description]
    - `dir_input` : [type]
        [description]
    - `sr_camera_params` : pd.Series
        [description]
    - `df_trajectory` : pd.DataFrame
        [description]
    - `df_corners` : pd.DataFrame
        [description]
    debug : int, optional
        [description], by default True
    dir_debug : Pathorstr, optional
        [description], by default None
    """    
    dir_calib_root = Path(dir_input)
    dir_debug = Path(dir_debug, "eval")

    K    = sr_camera_params.intrinsic.astype(float)
    dist = sr_camera_params.distortion.astype(float)
    size_img = (int(sr_camera_params.height), int(sr_camera_params.width)) 
    size_board_in_world_coord = board_aruco.getMarkerLength() * np.asarray(board_aruco.getGridSize()) +  board_aruco.getMarkerSeparation() * (np.asarray(board_aruco.getGridSize()) -1)

    cam = PinholeCamera(height=size_img[0], width=size_img[1], K=K)
    for name_img in df_trajectory.index:
        img = cv2.imread(str(Path(dir_calib_root, name_img)))
        img_out = img.copy()

        if np.isnan(df_corners.loc[name_img, "ids"]).any():
            pass
        else:
            cam.set_extrinsic_by_rtvec(
                rvec=df_trajectory.loc[name_img, "rvec"].astype(float), 
                tvec=df_trajectory.loc[name_img, "tvec"].astype(float)
            )
            img_out = cam.project_axis_on_image(unit_length=np.append(size_board_in_world_coord, 0.1), img=img_out, width_line=10)
        debug_vis(debug=debug, dir_debug=dir_debug, name_img=name_img, img_out=img_out)

    error = df_trajectory.loc[:, "error"].values.tolist() 
    error = np.asarray(error).reshape(-1)
    plt.bar( np.arange(1, len(error) + 1, dtype=np.int), error, label="Reprojection Error")
    plt.plot(np.arange(1, len(error) + 1, dtype=np.int), np.full_like(error, np.sqrt(np.average(error**2))), "--", color="r", label="RMS Error")
    plt.title("Calibrate Evaluation")
    plt.xlabel("index of image")
    plt.ylabel("error (pixel)")
    plt.legend()
    
    pth_output = dir_debug.parent / "calibrate_evaluation.png"
    plt.savefig(pth_output, dpi = 600)
    if debug:
        plt.show()
    ic(pth_output)
    return


def main(board_aruco, dir_input: Union[Path, str], size_img: Iterable[int]=None, debug=False, restart=True):
    dir_input  = Path(dir_input)
    dir_image  = Path(dir_input, "raw"  )
    dir_output = Path(dir_input, "data" )
    dir_debug  = Path(dir_input, "debug")

    size_img = cv2.imread(str(sorted(dir_image.glob("*.*"))[1])).shape[:2] if size_img is None else size_img

    pth_corners = Path(dir_output, "corners.pkl")
    if not pth_corners.exists():
    #if restart or not pth_corners.exists():
        df_corners = find_corners_aruco(
            dictionary_aruco=board_aruco.dictionary, 
            dir_input=dir_image, 
            debug=debug, dir_debug=dir_debug
        )
        pth_corners = save_corners(df_input=df_corners, dir_output=dir_output)
    else:
        df_corners = load_corners(pth_input=pth_corners)

    pth_camera_params = Path(dir_output, "camera_params.pkl")
    if restart or not pth_camera_params.exists():
        [sr_camera_params, df_trajectory] = calibrate_aruco(board_aruco=board_aruco, size_img=size_img, df_corners=df_corners)
        pth_camera_params = save_camera_parameters(df_input=sr_camera_params, dir_output=dir_output)
        pth_trajectory    = save_trajectory(df_input=df_trajectory, dir_output=dir_output)
        evaluatle(board_aruco=board_aruco, dir_input=dir_image, sr_camera_params=sr_camera_params, df_trajectory=df_trajectory, df_corners=df_corners, debug=debug, dir_debug=dir_debug)
    else:
        pass
    return [pth_camera_params, pth_trajectory, pth_corners]


if __name__ == "__main__":
    pth_calib_root = "/home/veily3/LIGAN/VeilyCV/test/test_508/calib"
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    board_aruco = aruco.GridBoard_create(markersX=13, markersY=9, markerLength=0.026, markerSeparation=0.0045, dictionary=aruco_dict)
    [pth_camera_params, pth_trajectory, pth_corners] = main(
        board_aruco=board_aruco,
        dir_input=pth_calib_root, debug=1
    )
    sr_camera_params  = load_camera_parameters(pth_camera_params)
    df_trajectory     = load_trajectory(pth_trajectory)
    df_coners         = load_trajectory(pth_corners)

    print()
