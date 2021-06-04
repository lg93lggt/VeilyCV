

from skimage import measure
import open3d as o3d
import numpy as np

def marching_cubes_skimage(voxels, grid):
    grid = grid.reshape((3, -1)).T 
    shape_voxels = voxels.shape

    voxels_padding = np.zeros((shape_voxels[0] + 2, shape_voxels[1] + 2, shape_voxels[2] + 2))
    voxels_padding[
        1: shape_voxels[0] + 1, 
        1: shape_voxels[1] + 1, 
        1: shape_voxels[2] + 1
    ] = voxels

    verts, faces, normals, values = measure.marching_cubes(voxels_padding)

    a = (grid.max(axis=0) - grid.min(axis=0))
    b = (verts.max(axis=0) - verts.min(axis=0))
    #verts = verts / b * a
    verts = verts * (a / voxels.shape)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.paint_uniform_color([1., 0, 0])
    return mesh