
import glob
import cv2
import numpy as np
import open3d as o3d
import time
import os
from icecream import ic
from src import Camera
from src.utils.debug import debug_vis


def estimate_grid(masks, intrinsic, extrinsics, resolution_new=200, z_constraint: np.ndarray=np.array([-np.inf, np.inf]), carve_outside=True, debug=False, dir_debug=None):

    resolution          = 50
    [x0, y0, z0]        = [-1, -1, -1] 
    [xlim, ylim , zlim] = [1, 1, 1] 
    [dx, dy, dz]        = (np.array([xlim, ylim ,zlim]) - np.array([x0, y0, z0])) / resolution

    grid = np.mgrid[x0:xlim:dx, y0:ylim:dy, z0:zlim:dz] 
    grid = grid.astype(np.float32)

    voxels_out = carving_with_probability(
        grid=grid, 
        masks=masks, 
        intrinsic=intrinsic, extrinsics=extrinsics, 
        carve_outside=carve_outside, 
        prop=0.8, 
        debug=debug, dir_debug=dir_debug
    )
    [X, Y, Z] = np.where(voxels_out)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((grid[:, X, Y, Z].reshape((3, -1))).T)

    center = pcd.get_center()
    ub = pcd.get_max_bound()
    lb = pcd.get_min_bound()
    l  = (ub - lb).max() + 0.1
    d  = l / resolution_new

    lz = max(center[2] - l/2, z_constraint[0])
    uz = min(center[2] + l/2, z_constraint[1])

    grid_new = np.mgrid[
        center[0]-l/2: center[0]+l/2: d,
        center[1]-l/2: center[1]+l/2: d,
        lz           : uz           : d,        
        ] 

    grid_new = grid_new.astype(np.float32)
    return grid_new


def carving_with_probability(grid, masks, intrinsic, extrinsics, carve_outside=True, prop=0.8, debug=False, dir_debug=None):
    eps = 1E-10

    grid_flatten = grid.reshape((3, -1))
    shape_voxels = grid.shape[1:]
    n_pts = grid_flatten.shape[1]

    [height_mask, width_mask] = masks[0].shape[:2]

    pts_homo = np.ones((4, n_pts)).astype(np.float32) # (4, n_points)
    pts_homo[:3] = grid_flatten

    voxels_prop = np.expand_dims(np.ones(n_pts), axis=0)
    voxels_prop = np.repeat(voxels_prop, len(masks), axis=0)
    for [cid, extrinsic] in enumerate(extrinsics):
        t0 = time.time()

        mask = masks[cid]
        ic(cid, "/", len(masks))

        if np.isnan(extrinsic).any():
            continue
        else:
            # Projection
            _p = intrinsic @ extrinsic @ pts_homo
            _p = _p / (_p[-1] + eps)
            pt2d = np.round(_p).astype(np.int)[:2].T

            bool_cols_in  = np.logical_and((0 <= pt2d[:, 0]), (pt2d[:, 0] <  width_mask ))
            bool_rows_in  = np.logical_and((0 <= pt2d[:, 1]), (pt2d[:, 1] <  height_mask))
            bool_within   = np.logical_and(bool_cols_in, bool_rows_in)

            [idxes_within] = np.where(bool_within)

            if carve_outside:
                [idxes_without] = np.where(np.logical_not(bool_within))
            pts2d_within = pt2d[idxes_within] # (n_cols, n_rows)
            mask_within =  mask[pts2d_within[:, 1], pts2d_within[:, 0]]

            [idxes_final] = np.where(mask_within==0)
            
            # Craving
            voxels_prop[cid, idxes_within[idxes_final]] = 0 # (1, n_points)
            if carve_outside:
                voxels_prop[cid, idxes_without] = 0

            t1 = time.time()
            ic(bool_within  .shape)
            ic(idxes_within .shape)
            if carve_outside:
                ic(idxes_without.shape)
            ic(pts2d_within .shape)
            ic(mask_within  .shape)
            ic(idxes_final  .shape)
            ic(len(np.where(voxels_prop[cid]!=0)[0]))
            dt = t1-t0
            ic(dt)
            print("*"*64)

        [X, Y, Z] = np.where(voxels_prop[cid].reshape(shape_voxels)==1)
        if np.isnan(extrinsics[cid]).any():
            continue
        else:
            cam = Camera.PinholeCamera(mask.shape[0], mask.shape[1], intrinsic)
            cam.set_extrinsic_matrix(extrinsics[cid])
            d = n_pts//10000 if (n_pts > 10000) else 1
            img_out = cam.project_points_on_image(points3d=grid[:, X, Y, Z].T[::d])
            img_out = cam.project_grid_on_image(grid, img=img_out, thickness=10)
            debug_vis(debug=debug, dir_debug=dir_debug, name_img="{:0>6d}.jpg".format(cid+1), img_out=img_out)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector((grid[:, X, Y, Z]* np.array([[[[1, 1, -1]]]]).reshape((3, 1))).T )
            # pcd
            # o3d.visualization.draw_geometries([pcd])

    judge = np.round(len(masks) * prop)
    voxels_out = voxels_prop.sum(axis=0)
    voxels_out = np.where(voxels_out<judge, 0, 1)
    voxels_out = voxels_out.reshape(shape_voxels).astype(bool)
    return voxels_out
