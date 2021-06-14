
import enum
import sys
import time
import cv2
from  icecream import ic
import numpy as np
import os
import pandas as pd
import glob
import open3d as o3d
from scipy import sparse
from matplotlib import cm
from pathlib import Path

sys.path.append("..")
from src import geometries
from pipelines.foot_3d_reconstruction.calibrate_camera_aruco import load_camera_parameters, load_trajectory
from src.utils.debug import debug_separator, debug_vis
from src.vl3d import voxel_craving, marching_cubes
from src.utils import plugins_labelme
from src import Camera


@ enum.unique
class ArucoCalibBoardType(enum.Enum):
    A3 = 0
    A4 = 1
ARUCO_BOARD_A3 = ArucoCalibBoardType.A3
ARUCO_BOARD_A4 = ArucoCalibBoardType.A4


@ debug_separator
def reconstructe_as_voxel_grid(dir_input, sr_camera_params, df_trajectory, grid=None, prop=0.8, z_constraint=np.array([-np.inf, np.inf]), carve_outside=True, debug=False, dir_debug=None):
    K = sr_camera_params.intrinsic.astype(float)

    df_isvalid = df_trajectory.loc[:, "error"].isna().apply(lambda x: not x)
    names_img_within = df_trajectory.loc[df_isvalid, :].index
    dir_mask = Path(dir_input, "mask")
    Ms = []
    masks = []
    for [_, name_img] in enumerate(names_img_within):
        pth_mask = Path(dir_mask, name_img)
        mask = cv2.imread(str(pth_mask), flags=cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    masks = np.array(masks)
    
    rvecs = np.asarray((df_trajectory.loc[df_isvalid, "rvec"].values).tolist(), dtype=float)
    tvecs = np.asarray((df_trajectory.loc[df_isvalid, "tvec"].values).tolist(), dtype=float)
    Ms = geometries.rtvecs_to_transform_matrices(rvecs, tvecs, shape=(3, 4))
    

    if grid is None:
        grid  =  voxel_craving.estimate_grid(
            masks=masks, 
            intrinsic=K, extrinsics=Ms, 
            resolution_new=200, 
            z_constraint=z_constraint, 
            debug=debug, dir_debug=os.path.join(dir_debug, "estimate_grid")
        )
    else:
        pass

    voxels = voxel_craving.carving_with_probability(
        grid, 
        masks=masks, 
        intrinsic=K, extrinsics=Ms, 
        carve_outside=carve_outside, 
        prop=prop,
        debug=debug, dir_debug=os.path.join(dir_debug, "voxel_craving")
    )

    dir_result = Path(dir_input, "result")
    dir_result.mkdir(parents=True, exist_ok=True)
    #allmatrix_sp = sparse.csr_matrix(voxels.reshape(-1)) # 采用行优先的方式压缩矩阵
    #ic(sparse.save_npz(os.path.join(dir_data, "voxel.npz"), allmatrix_sp)) # 保存稀疏矩阵
    
    [X, Y, Z] = np.where(voxels==1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((grid[:, X, Y, Z]* np.array([[[[1, 1, -1]]]]).reshape((3, 1))).T )

    ic(o3d.io.write_point_cloud(os.path.join(dir_result, "carving.ply"), pcd))

    if debug:
        o3d.visualization.draw_geometries([pcd])
    return [voxels, grid]


def applay_z_colormap(mesh, resolution_z=256):
    verts = np.asarray(mesh.vertices)
    zmax = verts[:, 2].max()
    zmin = verts[:, 2].min()
    dz   = (zmax - zmin) / resolution_z
    np.sort(verts[:, 2])
    colors_table = cm.jet(range(256))[:, :3]
    colors = np.zeros_like(verts)
    for i in  range(resolution_z):
        colors[zmin + i * dz < verts[:, 2], :] = colors_table[i*(256//resolution_z)]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3])
    return mesh


def main(dir_fit_root, dir_calib_root, size_img=None, prop=0.8, z_constraint=np.array([-0.05, np.inf]), aruco_board_type=None, debug=False):
    dir_debug         = Path(dir_fit_root, "debug", "voxel_grid")
    dir_image         = Path(dir_fit_root,  "raw")
    dir_result        = Path(dir_fit_root,  "result")
    pth_camera_params = Path(dir_calib_root, "data", "camera_params.pkl")
    pth_trajectory    = Path(dir_calib_root, "data", "trajectory.pkl")

    size_img = cv2.imread(str(sorted(dir_image.glob("*.*"))[0])).shape[:2] if size_img is None else size_img
    ic(size_img)

    """
    segment images by color in lab space
    """
    plugins_labelme.generate_masks(os.path.join(dir_fit_root, "labels"), save_mask=True, color=255)

    """
    Load camera parameters and trajectory frin pkl file
    """
    sr_camera = load_camera_parameters(pth_input=pth_camera_params)
    df_trajectory = load_trajectory(pth_input=pth_trajectory)

    """
    Estimate grid and Carving
    """
    if aruco_board_type == ARUCO_BOARD_A3:
        [x,y,z] = [0.4, 0.3, 0.2]
        grid = np.mgrid[0: x: x/200, 0: y: y/200, 0: z: z/200] 
        grid = grid.astype(np.float32)
    elif aruco_board_type == ARUCO_BOARD_A4:
        [x,y,z] = [0.3, 0.3, 0.2]
        grid = np.mgrid[-x: x: x/100, 0: y: y/200, 0: z: z/200] 
        grid = grid.astype(np.float32)
    else:
        grid = None
        
    [voxels, grid] = reconstructe_as_voxel_grid(
        dir_input=dir_fit_root, 
        grid=grid,
        sr_camera_params=sr_camera, 
        df_trajectory=df_trajectory, 
        prop=prop,
        z_constraint=z_constraint, 
        carve_outside=True, 
        debug=debug, dir_debug=dir_debug
    )
    if not voxels.any():
        ic(not voxels.any())
        return

    """
    Marching cubes to get mesh
    """
    mesh = marching_cubes.marching_cubes_skimage(voxels, grid)   
    mesh.compute_vertex_normals()
    mesh = mesh.filter_smooth_simple(number_of_iterations=10)
    mesh = applay_z_colormap(mesh)

    """
    Alignment mesh
    """
    obb = mesh.get_oriented_bounding_box()
    R = obb.R
    r = geometries.R_to_r(R)

    r_ = np.array([0, 0, r[0]])
    R_ = geometries.rvec_to_Rmat(r_, shape=(3, 3))
    mesh.rotate(R_, mesh.get_center())

    """
    IO
    """
    dir_result.mkdir(parents=True, exist_ok=True)
    pth_mesh_obj = Path(dir_result, "mesh.obj")
    pth_mesh_ply = Path(dir_result, "mesh.ply")
    o3d.io.write_triangle_mesh(str(pth_mesh_obj), mesh, write_vertex_colors=True, write_vertex_normals=True, print_progress=True)
    o3d.io.write_triangle_mesh(str(pth_mesh_ply), mesh, write_vertex_colors=True, write_vertex_normals=True, print_progress=True)
    ic(pth_mesh_obj)
    ic(pth_mesh_ply)

    if debug:
        o3d.visualization.draw_geometries([mesh])
    return mesh


if __name__ == '__main__':
    zc1 = np.array([-np.inf, 0])
    zc2 = np.array([-np.inf, -0.05])
    zc3= np.array([0, np.inf])
    dir_root       = "/home/veily3/LIGAN/VeilyCV/test/test_508/fit"
    dir_calib_root = "/home/veily3/LIGAN/VeilyCV/test/test_508/calib"
    main(dir_root, dir_calib_root, size_img=(1440, 1080), z_constraint=zc3, debug=10) 

