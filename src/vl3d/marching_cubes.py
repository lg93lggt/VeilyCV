

from skimage import measure
import open3d as o3d
import numpy as np

def marching_cubes_skimage(voxels, grid):
    grid = grid.reshape((3, -1)).T 
    shape_voxels = voxels.shape

    voxels_padding = np.zeros((shape_voxels[0] + 2, shape_voxels[1] + 2, shape_voxels[2] + 2))
    voxels_padding[
        1: -1, 
        1: -1, 
        1: -1
    ] = voxels

    verts, faces, normals, values = measure.marching_cubes(voxels_padding)
    verts = verts.astype(float)

    bound = (grid.max(axis=0) - grid.min(axis=0))
    
    verts = (verts - np.asarray(voxels_padding.shape) / 2) * ( bound / np.asarray(voxels_padding.shape)) + np.average(grid, 0) 

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.paint_uniform_color([1., 0, 0])
    return mesh


if __name__ == '__main__':
    [x,y,z] = [0.3, 0.3, 0.2]
    grid = np.mgrid[-x: x: x/100, 0: y: y/200, 0: z: z/200] 
    grid = grid.astype(np.float32)

    voxels = np.ones((200, 200, 200), dtype=bool)
    marching_cubes_skimage(voxels, grid)