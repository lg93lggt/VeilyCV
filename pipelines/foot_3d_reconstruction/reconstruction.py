
from sys import path
import cv2
from  icecream import ic

from scipy import sparse
from src import geometries
from src.pipelines.foot_3d_reconstruction.calibrate_dual import load_calib_dual
from src.pipelines.foot_3d_reconstruction.registration import load_trajectory
from src.pipelines.foot_3d_reconstruction.debug import separator
import numpy as np
import os
import pandas as pd
import glob
import open3d as o3d
from src.vl3d import voxel_craving, marching_cubes
from src.utils import plugins_labelme
from src import Camera



@ separator
def generate_voxel_grid(dir_calib_root, dir_fit_root, size_img, z_constraint=np.array([-np.inf, np.inf]), carve_outside=True, debug=False, dir_debug=None):
    dir_label = os.path.join(dir_fit_root, "labels")
    dir_data  = os.path.join(dir_fit_root, "data")

    [K, _, _] = load_calib_dual(dir_calib_root)
    traj = load_trajectory(dir_data)

    df_isvalid = traj.loc[:, "traj"].isna().apply(lambda x: not x)
    names_img_within = traj.loc[df_isvalid, "traj"].index

    Ms = []
    masks = []
    for [idx_img, name_img] in enumerate(names_img_within):
        if idx_img == 0:
            mask = np.ones(size_img).astype(np.float32)
            masks.append(mask)
            continue
        [prefix, _] = os.path.splitext(name_img)
        pth_label = os.path.join(dir_label, prefix + ".json")
        mask = plugins_labelme.generate_mask(pth_label, type_output=np.float32)
        masks.append(mask)
    
    Ms = np.asarray((traj.loc[df_isvalid, "traj"].values).tolist())
    Ms = Ms[:, :3, :] if Ms.shape[1] == 4 else Ms
    masks = np.array(masks)

    grid  =  voxel_craving.estimate_grid(
        masks=masks, 
        intrinsic=K, extrinsics=Ms, 
        resolution_new=200, 
        z_constraint=z_constraint, 
        debug=debug, dir_debug=os.path.join(dir_debug, "estimate_grid")
    )
    voxels = voxel_craving.carving_with_probability(
        grid, 
        masks=masks, 
        intrinsic=K, extrinsics=Ms, 
        carve_outside=carve_outside, 
        prop=1,
        debug=debug, dir_debug=os.path.join(dir_debug, "voxel_craving")
    )

    allmatrix_sp = sparse.csr_matrix(voxels.reshape(-1)) # 采用行优先的方式压缩矩阵
    ic(sparse.save_npz(os.path.join(dir_data, "voxel.npz"), allmatrix_sp)) # 保存稀疏矩阵
    
    [X, Y, Z] = np.where(voxels==1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((grid[:, X, Y, Z]* np.array([[[[1, 1, -1]]]]).reshape((3, 1))).T )

    ic(o3d.io.write_point_cloud(os.path.join(dir_data, "carving.ply"), pcd))

    if debug:
        o3d.visualization.draw_geometries([pcd])
    return [voxels, grid]


def main(dir_fit_root, dir_calib_root, size_img, z_constraint=np.array([-0.05, np.inf]), debug=False):

    """
    segment images by color in lab space
    """
    plugins_labelme.generate_masks(os.path.join(dir_fit_root, "labels"), save_mask=True, color=255)

    dir_debug = os.path.join(dir_fit_root, "debug", "voxel_grid")
    [voxels, grid] = generate_voxel_grid(dir_calib_root, dir_fit_root, size_img=size_img, z_constraint=z_constraint, carve_outside=True, debug=debug, dir_debug=dir_debug)
    mesh = marching_cubes.marching_cubes_skimage(voxels, grid)   
    mesh.compute_vertex_normals()
    mesh = mesh.filter_smooth_simple(number_of_iterations=10)

    pth_mesh = os.path.join(dir_fit_root, "data", "mesh_final.ply")
    o3d.io.write_triangle_mesh(pth_mesh, mesh)
    ic(pth_mesh)

    if debug:
        o3d.visualization.draw_geometries([mesh])
    return


if __name__ == '__main__':
    dir_root       = "/media/veily3/Data/ljj/foots_3d/dataset/fit"
    dir_calib_root = "/media/veily3/Data/ljj/foots_3d/dataset/calib"
    zc1 = np.array([-np.inf, 0])
    zc2 = np.array([-np.inf, -0.05])
    main(dir_root, dir_calib_root, size_img=(1440, 1080), z_constraint=zc2, debug=0) 

