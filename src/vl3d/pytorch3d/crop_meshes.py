

import pytorch3d.structures
import pytorch3d.ops
import open3d as o3d
import numpy as np

import sys
sys.path.append("../../../")
from src.utils.debug import debug_separator, debug_vis
from src.plugins.geometry3d import cvt_meshes_o3d_to_torch3d, cvt_meshes_torch3d_to_o3d

def crop_meshes_by_z(meshes: pytorch3d.structures.Meshes, z: float) -> pytorch3d.structures.Meshes:
    device = meshes.device
    meshes_o3d = cvt_meshes_torch3d_to_o3d(meshes)
    meshes_new: List[o3d.geometry.TriangleMesh] = []
    for mesh in meshes_o3d:
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox_new = o3d.geometry.AxisAlignedBoundingBox(
            bbox.min_bound, 
            np.array([
                bbox.max_bound[0], 
                bbox.max_bound[1], 
                bbox.min_bound[2] + z 
            ])
        )
        meshes_new.append(mesh.crop(bbox_new))
    meshes_cropped: pytorch3d.structures.Meshes = cvt_meshes_o3d_to_torch3d(meshes=meshes_new, device=device)
    return meshes_cropped