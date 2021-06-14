
import cv2
from pipelines.foot_3d_reconstruction.calibrate_camera_aruco import load_camera_parameters, load_trajectory   
from src import geometries
import numpy as np
from pathlib import Path

dir_root = Path("/home/veily3/LIGAN/VeilyCV/test/test_515/WC_L")

dir_root_dtu = Path(dir_root, "dtu")
dir_image_dtu = Path(dir_root_dtu, "image")
dir_mask_dtu = Path(dir_root_dtu, "mask")
dir_image_dtu.mkdir(exist_ok=True)
dir_mask_dtu.mkdir(exist_ok=True)

sr_camera = load_camera_parameters(Path(dir_root, "data/camera_params.pkl"))
df_traj   = load_trajectory(Path(dir_root, "data/trajectory.pkl"))

K = sr_camera.intrinsic.astype(float)

dtu_data = {}
for [i, name] in enumerate(df_traj[df_traj.rvec.notna()].index):
    rvec = df_traj.loc[name, "rvec"].astype(float)
    tvec = df_traj.loc[name, "tvec"].astype(float)
    P = K @ geometries.rtvec_to_transform_matrix(rvec, tvec, shape=(3, 4))
    dtu_data["world_mat_{:d}".format(i)] = P

    img = cv2.imread(str(Path(dir_root, "raw", name)))
    mask = cv2.imread(str(Path(dir_root, "mask", name)))
    cv2.imwrite(str(Path(dir_image_dtu, "{:0>3d}.png".format(i))), img)
    cv2.imwrite(str(Path(dir_mask_dtu, "{:0>3d}.png".format(i))), mask)


np.savez_compressed(Path(dir_root_dtu, "cameras_linear_init.npz"), **dtu_data)
a = np.load(Path(dir_root_dtu, "cameras_linear_init.npz"))
print()