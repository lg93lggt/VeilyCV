
import pytorch3d.structures
import pytorch3d.ops
from icecream import ic


def iterative_closest_point(meshes_src: pytorch3d.structures.Meshes, mesh_dst: pytorch3d.structures.Meshes) -> pytorch3d.structures.Meshes:
    """
    align meshes_src to mesh_dst
    """
    X = meshes_src.verts_padded()
    Y = mesh_dst.verts_padded().repeat(len(X), 1, 1)
    (converged, rmse, XT, _, _) = pytorch3d.ops.iterative_closest_point(X, Y)
    ic(converged, rmse)
    meshes_trans: pytorch3d.structures.Meshes = meshes_src.clone()
    meshes_trans.update_padded(XT)
    return meshes_trans