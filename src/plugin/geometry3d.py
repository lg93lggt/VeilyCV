
from typing import Iterable, List, Tuple
import numpy as np
import open3d as o3d
import pytorch3d.structures
import torch



def cvt_meshes_torch3d_to_o3d(meshes: pytorch3d.structures.Meshes) -> Iterable[o3d.geometry.TriangleMesh]:
    """[summary]

    [extended_summary]

    Parameters
    ----------
    meshes : pytorch3d.structures.Meshes
        [description]

    Returns
    -------
    Iterable[o3d.geometry.TriangleMesh]
        [description]
    """    
    meshes_o3d = []
    for i_mesh in range(len(meshes)):
        [verts, faces] = meshes.get_mesh_verts_faces(i_mesh)
        mesh_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts.cpu().numpy()),
            o3d.utility.Vector3iVector(faces.cpu().numpy()),
        )
        meshes_o3d.append(mesh_o3d)
    return meshes_o3d


def cvt_mesh_o3d_to_torch3d(mesh: o3d.geometry.TriangleMesh, device="cuda:0") -> pytorch3d.structures.Meshes:
    """---
    cvt_mesh_o3d_to_torch3d
    [extended_summary]

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        [triangle mesh in o3d format]
    device : str, optional
        [description], by default "cuda:0"

    Returns
    -------
    pytorch3d.structures.Meshes
        [description]
    """    
    v = np.asarray(mesh.vertices).astype(np.float32)
    f = np.asarray(mesh.triangles).astype(int)
    verts_padded = torch.from_numpy(v).unsqueeze(0).to(device=device)
    faces_padded = torch.from_numpy(f).unsqueeze(0).to(device=device)
    meshes_torch3d = pytorch3d.structures.Meshes(verts=verts_padded, faces=faces_padded)
    meshes_torch3d = meshes_torch3d.to(device)
    return meshes_torch3d


def cvt_meshes_o3d_to_torch3d(meshes: Iterable[o3d.geometry.TriangleMesh], device="cuda:0") -> pytorch3d.structures.Meshes:
    """[summary]

    [extended_summary]

    Parameters
    ----------
    meshes : Iterable[o3d.geometry.TriangleMesh]
        [description]
    device : str, optional
        [description], by default "cuda:0"

    Returns
    -------
    pytorch3d.structures.Meshes
        [description]
    """    
    verts_list = []
    faces_list = []
    for mesh in meshes:
        v = np.asarray(mesh.vertices).astype(np.float32)
        f = np.asarray(mesh.triangles).astype(int)
        verts_list.append(torch.from_numpy(v))
        faces_list.append(torch.from_numpy(f))
    meshes_torch3d = pytorch3d.structures.Meshes(verts=verts_list, faces=faces_list)
    meshes_torch3d = meshes_torch3d.to(device)
    return meshes_torch3d


def cvt_mesh_o3d_to_vl3d(mesh: o3d.geometry.TriangleMesh):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        [description]
    """    
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    self.vbo = vbo.VBO(verts.flatten())
    self.ebo = vbo.VBO(faces.flatten(), target = GL_ELEMENT_ARRAY_BUFFER)
    
    self.eboLength = faces.size
    self.bCreate = True
    return