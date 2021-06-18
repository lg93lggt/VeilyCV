

import enum
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
import pytorch3d.io
import pytorch3d.loss
import pytorch3d.ops
import pytorch3d.renderer
import pytorch3d.structures
import pytorch3d.transforms
import torch
import torch.nn
from icecream import ic
from src import geometries
from src.Camera import PinholeCamera




""" NOTE
image point (u, v) to ndc point (ndcx, ndcy)
u: (0, w -1) -> (+1, -1) 
v: (0, h -1) -> (+1, -1) 
"""

@ enum.unique
class PLUGIN_PYTORCH3D_CAMERA(enum.Enum):
    """---
    Format of pytorch3d camera intrinsic matrix 
    """
    PROJECTION_MATRIX = enum.auto()
    K = enum.auto()


def cvt_intrinsic_matrix_cv2_to_torch3d(
    K: np.ndarray, 
    size_image: Tuple[int], 
    n_batch: int = 1, 
    flags: PLUGIN_PYTORCH3D_CAMERA = PLUGIN_PYTORCH3D_CAMERA.K, 
    device: str = "cuda:0"
) -> torch.Tensor:
    r"""---
    # cvt_intrinsic_matrix_cv2_to_torch3d
    Convert intrinsic matrix K from OpenCV format to pytorch3d format

    NOTE: 
    In OpenCV   : P_IMG ~ (K1 @ T @ R @ X).T
    In Pytorch3d: P_NDC ~ (K2.T @ T @ R @ X).T = X.T @ R2 @ T2 @ K2
    
    Parameters
    -------
    ### - `K`:
    shape=(4, 4), camera intrinsic matrix in OpenCV camera format   
    ### - `size_image`: 
        (height, width), image size in cv2 format
    ### - `n_batch` (default=1): 
        num of batch in output camera intrinsic matrix _K
    ### - `flags` (default=PLUGIN_PYTORCH3D_CAMERA.K): 
    - PLUGIN_PYTORCH3D_CAMERA.K
    - PLUGIN_PYTORCH3D_CAMERA.PROJECTION_MATRIX
        
    Returns
    -------
    ### - `_K`: 
        shape=(n_batch, 4, 4), output camera intrinsic matrix in pytorch3d format
        format PerspectiveCameras.K if flags==PLUGIN_PYTORCH3D_CAMERA.K
        format PerspectiveCameras.get_projection_transform().get_matrix() if flags==PLUGIN_PYTORCH3D_CAMERA.PROJECTION_MATRIX

    Raises
    ------
    - KeyError:   Using unspportted flags.
    - ValueError: Shape of K should be (4, 4).
    """
    if K.shape != (4, 4):
        raise ValueError("Shape of K should be (4, 4).")
    
    K = torch.from_numpy(K).to(dtype=torch.float32, device=device).unsqueeze(0)
    
    sh = - 2 / (size_image[0] - 1)
    sw = - 2 / (size_image[1] - 1)
    S = torch.tensor([
        [sw,  0, 1, 0],
        [ 0, sh, 1, 0],
        [ 0,  0, 0, 1],
        [ 0,  0, 1, 0],
    ]).to(dtype=torch.float32, device=device)
    _K = S @ K
    if flags == PLUGIN_PYTORCH3D_CAMERA.K:
        _K = _K
    elif flags == PLUGIN_PYTORCH3D_CAMERA.PROJECTION_MATRIX:
        _K = _K.permute(0, 2, 1)
    else:
        raise KeyError("Using unspportted flags in Parameters.")
    _K = _K.repeat(n_batch, 1, 1)
    return _K


def cvt_intrinsic_matrix_torch3d_to_cv2(
    K: torch.Tensor, 
    size_image: torch.Tensor, 
    flags: PLUGIN_PYTORCH3D_CAMERA = PLUGIN_PYTORCH3D_CAMERA.K
) -> np.ndarray:
    r"""---
    # cvt_intrinsic_matrix_cv2_to_torch3d
    Convert intrinsic matrix K from OpenCV format to pytorch3d format
    
        NOTE: 
        In OpenCV   : P_IMG ~ (K1 @ T @ R @ X).T
        In Pytorch3d: P_NDC ~ (K2.T @ T @ R @ X).T = X.T @ R2 @ T2 @ K2
    
    Parameters
    -------
    
    ### - `K`:
        shape=(1, 4, 4), camera intrinsic matrix in pytorch3d camera format
        
    ### - `size_image`: 
        shape=(1, width, height), image size in pytorch3d camera format
    
    ### - `flags` (default=PLUGIN_PYTORCH3D_CAMERA.K): 
    
        1. PLUGIN_PYTORCH3D_CAMERA.K
        
        input K is PerspectiveCameras.K
        
        2. PLUGIN_PYTORCH3D_CAMERA.PROJECTION_MATRIX
        
        input K is PerspectiveCameras.get_projection_transform().get_matrix()

    Returns
    -------
    ### - `_K`: 
        shape=(4, 4), output camera intrinsic matrix in OpenCV format

    Raises
    ------
    - KeyError: Using unspportted flags in Parameters.
    """
    device = K.device
    
    sw_i = - (size_image[0, 0] - 1) / 2
    sh_i = - (size_image[0, 1] - 1) / 2
    SI = torch.tensor([
        [sw_i,     0, 0, -sw_i],
        [   0 , sh_i, 0, -sh_i],
        [   0,     0, 0,     1],
        [   0,     0, 1,     0],
    ]).to(dtype=torch.float32, device=device)
    if flags == PLUGIN_PYTORCH3D_CAMERA.K:
        K = K
    elif flags == PLUGIN_PYTORCH3D_CAMERA.PROJECTION_MATRIX:
        K = K.permute(0, 2, 1)
    else:
        raise KeyError("Using unspportted flags in Parameters.")
    _K = SI @ K
    return _K.cpu().numpy()


def cvt_size_image_cv2_to_torch3d(size_image: Iterable[int], n_batch: int = 1, device: str = "cuda:0") -> torch.Tensor:
    r"""---
    # cvt_size_image_cv2_to_torch3d
    Convert image size from OpenCV format to pytorch3d format
    
        NOTE: 
        In OpenCV   : (height, width)
        In Pytorch3d: (width, height)
    Parameters
    -------
    
    ### - `size_image`:
        (height, width) | (height, width, channel), input image size in OpenCV format
        
    ### - `n_batch` (default=1): 
        num of batch in output camera image size 
    
    Returns
    -------
    ### - `size_image_torch`: 
        shape=(n_batch, 2), output image size in pytorch3d camera format

    Raises
    ------
    - ValueError: Length of size_image should be 2 | 3.
    """
    if len(size_image) not in [2, 3]:
        raise ValueError("Length of size_image should be 2 | 3.")
    size_image = np.asarray(size_image)
    size_image = np.flip(size_image)
    size_image_torch = torch.from_numpy(size_image.copy()) \
        .to(dtype=torch.float32, device=device) \
        .unsqueeze(0) \
        .repeat(n_batch, 1)
    return size_image_torch


def cvt_rotation_matrix_cv2_to_torch3d(R: np.ndarray, n_batch: int = 1, device: str = "cuda:0") -> torch.Tensor:
    r"""---
    # cvt_rotation_matrix_cv2_to_torch3d
    Convert rotation matrix R from OpenCV format to pytorch3d format  
      
    _R = R.T
    
    Parameters
    -------
    
    ### - `R`:
        shape=(3, 3), input rotation matrix of SO3 in OpenCV format
        
    ### - `n_batch`: 
        num of batch in output rotation matrix _R
    
    ### - `device` (default="cuda:0"): 
        device of torch

    Returns
    -------
    ### - `_R`: 
        shape=(n_batch, 3, 3), output rotation matrix in pytorch3d format

    Raises
    ------
    - ValueError: Shape of rotation matrix R should be (3, 3).
    """
    if R.shape != (3, 3):
        raise ValueError("Shape of rotation matrix R should be (3, 3).")
    _R = torch.from_numpy(R.T) \
        .unsqueeze(0) \
        .repeat(n_batch, 1, 1) \
        .to(dtype=torch.float32, device=device)
    return _R


def cvt_translation_vector_cv2_to_torch3d(t: np.ndarray, n_batch: int = 1, device: str = "cuda:0") -> torch.Tensor:
    r"""---
    # cvt_rotation_matrix_cv2_to_torch3d
    Convert translation vector t from OpenCV format to pytorch3d format  
      
    _t = t
    
    Parameters
    -------
    ### - `t`:
        shape=(3, ) | (1, 3) | (3, 1), input translation vector in OpenCV format
    ### - `n_batch`: 
        num of batch in output translation vector _t
    ### - `device` (default="cuda:0"): 
        device of torch

    Returns
    -------
    ### - `_t`: 
        shape=(n_batch, 3), output rotation matrix in pytorch3d format

    Raises
    ------
    - ValueError: Shape of translation vector t should be (3, ) | (1, 3) | (3, 1).
    """
    if t.size != 3:
        raise ValueError("Shape of translation vector t should be (3, ) | (1, 3) | (3, 1).")
    t = t.flatten()
    _t = torch.from_numpy(t) \
        .unsqueeze(0) \
        .repeat(n_batch, 1) \
        .to(dtype=torch.float32, device=device)
    return _t


def cvt_extrinsic_matrix_cv2_to_torch3d(M: np.ndarray, n_batch: int = 1, device: str = "cuda:0") -> Tuple[torch.Tensor]:
    r"""---
    # cvt_extrinsic_matrix_cv2_to_torch3d
    Convert transform matrix M from OpenCV format to pytorch3d format  
      
    [R | t] = M, _R = R.T, _T = t
    
    Parameters
    -------
    ### - `M`:
        shape=(3, 4) | (4, 4), input transform matrix in OpenCV format 
    ### - `n_batch`: 
        num of batch in output translation vector _t and output rotation matrix _R
    ### - `device` (default="cuda:0"): 
        device of torch

    Returns
    -------
    ### - `_R`: 
        shape=(n_batch, 3, 3), output rotation matrix in pytorch3d format
    ### - `_t`: 
        shape=(n_batch, 3), output rotation matrix in pytorch3d format

    Raises
    ------
    - ValueError: Shape of rotation    matrix R should be (3, 3).
    - ValueError: Shape of translation vector t should be (3, ) | (1, 3) | (3, 1).
    """
    (R, t) = geometries.decompose_transform_matrix_to_Rmat_tvec(M)
    _R = cvt_rotation_matrix_cv2_to_torch3d(R, n_batch=n_batch, device=device)
    _t = cvt_translation_vector_cv2_to_torch3d(t, n_batch=n_batch, device=device)
    return (_R, _t)


def cvt_camera_vl_to_torch3d(camera: PinholeCamera, n_batch: int = 1, device: str = "cuda:0") -> pytorch3d.renderer.cameras.PerspectiveCameras:
    r"""---
    # cvt_camera_vl_to_torch3d
    Convert camera in VeilyCV format to cameras pytorch3d format

    Parameters
    -------
    ### - `camera`:
        camera in VeilyCV format    
    ### - `n_batch`: 
        num of batch in output cameras in pytorch3d format
    ### - `device` (default="cuda:0"): 
        device of torch

    Returns
    -------
    ### - `cameras_torch`: 
        size=(n_batch), output cameras in pytorch3d format
        
    Raises
    ------
    - ValueError: Shape of rotation    matrix R should be (3, 3).
    - ValueError: Shape of translation vector t should be (3, ) | (1, 3) | (3, 1).
    """
    K = cvt_intrinsic_matrix_cv2_to_torch3d(
        K=camera.intrinsic._mat_4x4(), 
        size_image=camera.size_image, 
        n_batch=n_batch,
        flags=PLUGIN_PYTORCH3D_CAMERA.K, 
        device=device
    )
    size_image_torch = cvt_size_image_cv2_to_torch3d(
        camera.size_image, 
        n_batch=n_batch, 
        device=device
    )
    (R, T) = cvt_extrinsic_matrix_cv2_to_torch3d(
        camera.extrinsic._mat_4x4(),
        n_batch=n_batch,
        device=device
    )
    cameras_torch = pytorch3d.renderer.cameras.PerspectiveCameras(
        K=K,
        R=R,
        T=T,
        image_size=size_image_torch,
        device=device,
    )
    return cameras_torch


def cvt_camera_vl_with_trajectory_to_torch3d(camera: PinholeCamera, idxes: Union[Iterable[int], None] = None , device: str = "cuda:0") -> pytorch3d.renderer.cameras.PerspectiveCameras:
    r"""---
    # cvt_camera_vl_with_trajectory_to_torch3d
    Convert camera with trajectory in VeilyCV format to cameras pytorch3d format

    Parameters
    -------
    ### - `camera`:
        camera in VeilyCV format
    ### - `device` (default="cuda:0"): 
        device of torch
    ### - `idxes` (default=None):
        select index in trajectory from List idxes
        select all when idxes is None
        
    Returns
    -------
    ### - `cameras_torch`: 
        size=(n_trajectory), output cameras in pytorch3d format
        
    Raises
    ------
    - ValueError: Shape of rotation    matrix R should be (3, 3).
    - ValueError: Shape of translation vector t should be (3, ) | (1, 3) | (3, 1).
    """
    n_traj = len(camera.trajectory)
    n_camera = n_traj if idxes is None else len(idxes)
    trajectory = camera.trajectory if idxes is None else camera.trajectory[idxes]

    K = cvt_intrinsic_matrix_cv2_to_torch3d(
        K=camera.intrinsic._mat_4x4(), 
        size_image=camera.size_image, 
        device=device
    )
    K = K.repeat((n_camera, 1, 1)).to(dtype=torch.float32, device=device)

    R = torch.zeros((n_camera, 3, 3)).to(dtype=torch.float32, device=device)
    T = torch.zeros((n_camera, 3)).to(dtype=torch.float32, device=device)
    image_size = torch.zeros((n_camera, 2)).to(
        dtype=torch.float32, device=device)
    for [i_traj, M] in enumerate(trajectory):
        image_size[i_traj] = torch.from_numpy(
            np.asarray(camera.size_image).astype(np.float32))
        (R_tmp, T_tmp) = cvt_extrinsic_matrix_cv2_to_torch3d(M, device=device)
        R[i_traj] = R_tmp.to(dtype=torch.float32, device=device)
        T[i_traj] = T_tmp.to(dtype=torch.float32, device=device)
    image_size = torch.flip(image_size, dims=[1])
    cameras_torch = pytorch3d.renderer.cameras.PerspectiveCameras(
        K=K, R=R, T=T, image_size=image_size, device=device)
    return cameras_torch


if __name__ == "__main__":
    
    pth1 = Path("/media/veily3/data_ligan/voxel_carving_data/project/ligan_right_0526-1317/data/camera_params.pkl")
    pth2 = pth1.parent / "trajectory.pkl"
    cam = PinholeCamera(360, 270)
    cam.load_from_file(pth1)
    cam.load_from_file(pth2)
    
    cams_torch = cvt_camera_vl_with_trajectory_to_torch3d(cam)
    for i in range(len(cam.names_traj)):
        print("-"*100)
        ic(i)
        cam.apply_trajectory_by_index(i)

        points_world_cv = np.array([
            [0   , 0   , 0   , 1], 
            [0.18, 0   , 0   , 1], 
            [0   , 0.18, 0   , 1], 
            [0   , 0   , 0.18, 1],
        ])
        points_world_torch = torch.from_numpy(points_world_cv).to(
            dtype=torch.float32, device="cuda:0").unsqueeze(0)

        # (w, h) in torch
        # (h, w) in cv
        size_image_torch = torch.tensor([[cam.width, cam.height]]).to(
            dtype=torch.float32, device="cuda:0")

        P_world_cv = (cam.extrinsic._mat_4x4() @ points_world_cv.T).T
        P_world_torch = points_world_torch @ cams_torch.get_world_to_view_transform().get_matrix()[i]

        # CHECK
        loss_extrinsic = torch.dist(
            P_world_torch, 
            torch.from_numpy(P_world_cv).to(dtype=torch.float32, device="cuda:0").unsqueeze(0)
        )
        ic(loss_extrinsic)

        """
        image point (u, v) to ndc point (ndcx, ndcy)
        u: (0, w -1) -> (+1, -1) 
        v: (0, h -1) -> (+1, -1) 
        """
        pts_cv_img_gt = cam.project_points(
            points_world_torch[0, :, :3].cpu().numpy())
        ndcx = 1 - 2 * (pts_cv_img_gt[:, 0]) / (cam.width - 1)
        ndcy = 1 - 2 * (pts_cv_img_gt[:, 1]) / (cam.height - 1)
        ndc_cv_gt = np.vstack((ndcx, ndcy)).T

        # solve fx fy cx cy in pytorch3d
        A = np.zeros((2*pts_cv_img_gt.shape[0], 4))
        A[0::2, 0] = P_world_cv[:, 0] / P_world_cv[:, 2]
        A[1::2, 1] = P_world_cv[:, 1] / P_world_cv[:, 2]
        A[0::2, 2] = 1
        A[1::2, 3] = 1
        b = ndc_cv_gt.flatten().reshape((-1, 1))
        (X, _, _, _) = np.linalg.lstsq(A, b, rcond=None)
        X = X.flatten()

        K_torch_gt = torch.tensor([
            [X[0],    0, X[2], 0],
            [   0, X[1], X[3], 0],
            [   0,    0,    0, 1],
            [   0,    0,    1, 0]
        ]).to(dtype=torch.float32, device="cuda:0").unsqueeze(0)

        """
        build torch camera
        """
        (R, T) = cvt_extrinsic_matrix_cv2_to_torch3d(cam.extrinsic._mat_4x4())
        cam_torch_gt = pytorch3d.renderer.PerspectiveCameras(
            K=K_torch_gt, R=R, T=T, image_size=size_image_torch, device="cuda:0")
        pts_torch_ndc = cam_torch_gt.transform_points(
            points_world_torch[..., :3])
        pts_torch_img = cam_torch_gt.transform_points_screen(
            points_world_torch[..., :3], size_image_torch)

        # CHECK
        loss_ndc = torch.dist(pts_torch_ndc[..., :2], torch.from_numpy(
            ndc_cv_gt).to(dtype=torch.float32, device="cuda:0").unsqueeze(0))
        ic(loss_ndc)
        # CHECK
        loss_projection = torch.dist(pts_torch_img[..., :2], torch.from_numpy(
            pts_cv_img_gt).to(dtype=torch.float32, device="cuda:0").unsqueeze(0))
        ic(loss_projection)

        K0 = cam.intrinsic._mat_4x4()
        K1 = cam_torch_gt.K[0]
        K2 = cvt_intrinsic_matrix_cv2_to_torch3d(
            K0, size_image=cam.size_image)
        K3 = cvt_intrinsic_matrix_torch3d_to_cv2(
            cams_torch.K[i], size_image=cams_torch.image_size)

        # CHECK
        loss_K = (torch.dist(K1, K2), np.linalg.norm(K0 - K3))
        ic(loss_K)
