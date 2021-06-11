
from typing import *
import numpy as np
import cv2
from easydict import EasyDict
import scipy 



def to_homo(P, axis: int=0):
    n_dims = len(P.shape)
    if axis == 0:
        if n_dims == 1:
            _P = np.hstack((P, 1)).reshape((-1, 1))
            return _P
        elif n_dims == 2:
            _P = np.hstack(
                (P, np.ones((P.shape[0], 1)))
            ).T
            return _P
        else:
            raise IndexError("Unknown Dimension: len({})={:d}".format(P.shape, n_dims))
    elif axis == 1:
            _P = np.vstack(
                (P, np.ones((1, P.shape[0])))
            )
            return _P
    else :
        raise IndexError("Unknown Main Axis: axis={:d}".format(axis))

def distance_SO3(R1, R2):
    d12 = np.linalg.norm(scipy.linalg.logm(R1.T @ R2)) / np.sqrt(2)
    d21 = np.linalg.norm(scipy.linalg.logm(R2.T @ R1)) / np.sqrt(2)
    dist = (d12 + d21) / 2
    return dist

def distance_list_SO3(Rs1, Rs2):
    num  = Rs1.shape[0]
    dist = np.zeros(num)
    for idx in range(num):
        R1 = Rs1[idx]
        R2 = Rs2[idx]
        dist[idx] = distance_SO3(R1, R2)
    return np.average(dist)

def average_rvecs(rvecs):
    n_vecs = rvecs.shape[0]

    R_avg = np.zeros((3, 3))
    for rvec in rvecs:
        R = r_to_R(rvec)[:3, :3]
        R_avg += scipy.linalg.logm(R)
    R_avg = R_avg / n_vecs
    R_avg = scipy.linalg.expm(R_avg)
    rvec_avg = R_to_r(R_avg)
    return rvec_avg


def average_SO3(Rs):
    n_mats = Rs.shape[0]

    R_avg = np.zeros((3, 3))
    for R in Rs:
        R_avg += scipy.linalg.logm(R)
    R_avg = R_avg / n_mats
    R_avg = scipy.linalg.expm(R_avg)
    return R_avg
        
def average_SE3(Ms):
    rs, ts = decompose_transform_matrices_to_rtvecs(Ms)
    Rs = rs_to_Rs(rs, shape=(3, 3))
    [r_avg, _] = cv2.Rodrigues(average_SO3(Rs))

    t_avg = np.average(ts, axis=0)

    M_avg = rtvec_to_transform_matrix(rvec=r_avg, tvec=t_avg)
    return M_avg

def rtvec_to_transform_matrix(rvec=np.zeros(3), tvec=np.zeros(3), shape=(4, 4)):
    M = np.eye(4)
    [R, _] = cv2.Rodrigues(rvec)
    M[:3, :3] = R
    M[:3,  3] = tvec.flatten()
    if   shape == (4, 4):
        return M
    elif shape == (3, 4):
        return M[:3]
    else:
        raise IndexError("Unsuppoert Matrix Shape: {}, Which Should Be (4, 4) or (3, 4)".format(shape))


def rtvecs_to_transform_matrixes(rvecs=np.zeros((1, 3)), tvecs=np.zeros((1, 3)), shape=(4, 4)):
    M = np.eye(4)
    if isinstance(rvecs, List):
        rvecs = np.array(rvecs)
    if isinstance(tvecs, List):
        tvecs = np.array(tvecs)
    n_rvecs = rvecs.shape[0]
    n_tvecs = rvecs.shape[0]
    if n_rvecs != n_tvecs:
        raise IndexError("Nuequal Num of rvecs and tvecs.")
    else:
        Ms = np.expand_dims(np.eye(4),0).repeat(n_rvecs,axis=0)
        for i_vec in range(n_rvecs):
            rvec = rvecs[i_vec]
            tvec = tvecs[i_vec]

            [R, _] = cv2.Rodrigues(rvec)
            Ms[i_vec, :3, :3] = R
            Ms[i_vec, :3,  3] = tvec.flatten()
        if   shape == (4, 4):
            return Ms
        elif shape == (3, 4):
            return Ms[:, :3]
        else:
            raise IndexError("Unsuppoert Matrix Shape: {}, Which Should Be (4, 4) or (3, 4)".format(shape))


def transform3d(M, P, axis: int=0, is_homo: bool=True):
    P = to_homo(P, axis=axis)
    _P = M @ P
    return _P


def Rz(theta_z, in_degree=False):
    if in_degree:
        theta_z = np.deg2rad(theta_z)
    Rz = np.eye(4)
    Rz[:2, :2] = np.array([
        [ np.cos(theta_z), np.sin(theta_z), ],
        [-np.sin(theta_z), np.cos(theta_z), ],
    ])
    return Rz


def R_to_r(R: np.ndarray)-> np.ndarray:
    """
        旋转矩阵转向量
    """
    R_ = R[:3, :3]
    rvec = cv2.Rodrigues(R_)[0].flatten()
    return rvec


def r_to_R(rvec: np.ndarray, shape=(4, 4))-> np.ndarray:
    """
        旋转向量转矩阵
    """
    R = np.eye(4)
    R_3x3 = cv2.Rodrigues(rvec)[0]
    R[:3,  :3] = R_3x3
    if   shape==(4, 4):
        return R
    elif shape==(3, 3):
        return R_3x3
    else:
        raise IndexError("shape should be 3x3 or 4x4")

def rs_to_Rs(rvecs: np.ndarray, shape=(4, 4))-> np.ndarray:
    """
        旋转向量转矩阵
    """
    n = rvecs.shape[0]
    if   shape==(4, 4):
        Rs = np.zeros((n, 4, 4))
    elif shape==(3, 3):
        Rs = np.zeros((n, 3, 3))
    else:
        raise IndexError("shape should be 3x3 or 4x4")
    
    for [i, rvec] in enumerate(rvecs):
        R = r_to_R(rvec, shape=shape)
        Rs[i] = R
    return Rs

def T_to_t(T: np.ndarray)-> np.ndarray:
    """
        平移矩阵转向量
    """
    tvec = T[:3, 3]
    return tvec

def t_to_T(tvec: np.ndarray)-> np.ndarray:
    """
        平移向量转矩阵
    """
    if tvec.size == 3:
        tvec = tvec.flatten()
    T = np.eye(4)
    T[:3, 3] = tvec
    return T

def decompose_transform_matrix_to_RTmat(M: np.ndarray):
    """
    Decompse Transform matrix to: 
        @R mat, shape: (4, 4)
        @T mat, shape: (4, 4)
    """
    R = np.eye(4)
    T = np.eye(4)
    R[:3, :3] = M[:3, :3]
    T[:3,  3] = M[:3,  3]
    return (R, T)

def decompose_transform_matrix_to_Rmat_tvec(M: np.ndarray):
    """
    Decompse Transform matrix to: 
        @R mat, shape: (3, 3)
        @t vec, shape: (1, 3)
    """
    (R, T) = decompose_transform_matrix_to_RTmat(M)
    R = R[:3, :3]
    t = T[:3, -1]
    return (R, t)

def decompose_transform_matrix_to_rtvec(M: np.ndarray) -> np.ndarray:
    """
        位姿矩阵转向量
    """
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    R = np.eye(4)
    R[:3, :3] = M[:3, :3]
    T = np.eye(4)
    T[:3,  3] = M[:3,  3]
    rvec = R_to_r(R)
    tvec = T_to_t(T)
    return [rvec, tvec]


def decompose_transform_matrices_to_rtvecs(Ms: np.ndarray) -> np.ndarray:
    """
        位姿矩阵转向量
    """
    n_veces = Ms.shape[0]
    rvecs = np.zeros((n_veces, 3))
    tvecs = np.zeros((n_veces, 3))
    for (idx_vec, M) in enumerate(Ms):
        [rvecs[idx_vec], tvecs[idx_vec]] = decompose_transform_matrix_to_rtvec(M)

    return [rvecs, tvecs]


def transform_matrix_inverse(M: np.ndarray) -> np.ndarray:
    """
        逆
    """
    (R, T) = decompose_transform_matrix_to_RTmat(M)
    tvec = T_to_t(T)
    RI = R.T
    cvec = -RI[:3, :3] @ tvec
    C = t_to_T(cvec)
    MI = C @ RI 
    return MI

def rtvec_to_rtmat(rtvec: np.ndarray) -> np.ndarray:
    """
        位姿向量转矩阵
    """
    rtvec = rtvec.reshape(6)
    R = r_to_R(rtvec[:3])
    T = t_to_T(rtvec[3:])
    return T @ R


def rtvec_degree2rad(rtvec_degree: np.ndarray) -> np.ndarray:
    """
        rtvec角度转弧度
    """
    rtvec_rad = rtvec_degree.copy()
    rtvec_rad[:3] = np.pi * (rtvec_rad[:3] / 180)
    return rtvec_rad
    

def rtvec_rad2degree(rtvec_rad: np.ndarray) -> np.ndarray:
    """
        rtvec弧度转角度
    """
    rtvec_degree = rtvec_rad.copy()
    rtvec_degree[:3] = 180 * (rtvec_degree[:3] / np.pi)
    return rtvec_degree


def solve_projection_matrix_3d_to_2d(points3d: np.ndarray, points2d: np.ndarray, method="svd")-> np.ndarray:
    """
        解3d-2d投影矩阵
        SVD或OLS方法求解
    """
    n_points3d = points3d.shape[0]
    n_points2d = points2d.shape[0]
    if n_points3d != n_points2d:
        raise IndexError
    else:
        n_points = n_points3d

    # format equation: Am = b
    A = np.zeros((2 * n_points, 11))
    b = np.zeros( 2 * n_points ) 
    for idx_point in range(n_points):
        point3d = points3d[idx_point]
        point2d = points2d[idx_point]

        x = point3d[0]
        y = point3d[1]
        z = point3d[2]
        u = point2d[0]
        v = point2d[1]

        #debug 
        # print("x: {:3f}, y: {:3f}, z: {:3f}, u: {:3f}, v: {:3f}".format(x,y,z,u,v))

        A[idx_point*2    , :] = np.array([x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z])
        A[idx_point*2 + 1, :] = np.array([0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z])
        b[idx_point*2       ] = u
        b[idx_point*2 + 1   ] = v
    #debug print(A, "\n", b)

    if  method == "ols":
        M = np.eye(4)
        m = np.linalg.lstsq(A, b, rcond=None)[0]
        M[:3, :] = np.reshape(np.append(m, 1), (3, 4))
        return M

    elif method == "svd":
        N = np.eye(4)
        # format equation: Cn = 0
        C = np.hstack((A, -b.reshape((n_points * 2, 1))))
        _, _, VT = np.linalg.svd(C)
        n = VT[-1, :]
        N[:3, :] = n.reshape((3, 4))
        return N
    else:
        raise TypeError

def decompose_projection_mat(mat_projection: np.ndarray):
    """
        分解投影矩阵
        公式法, 旋转矩阵不一定保证正交
    """
    M_ = mat_projection
    m34 = 1 / np.linalg.norm(M_[2, :3])
    M = M_ * m34
    
    m1 = M[0, :3]
    m2 = M[1, :3]
    m3 = M[2, :3]


    fx = np.linalg.norm(np.cross(m1, m3))
    fy = np.linalg.norm(np.cross(m2, m3))

    cx = np.dot(m1, m3)
    cy = np.dot(m2, m3)

    r1 = (m1 - cx*m3) / fx
    r2 = (m2 - cy*m3) / fy
    r3 = m3

    t1 = (M[0, 3] - cx*M[2, 3]) / fx
    t2 = (M[1, 3] - cy*M[2, 3]) / fy
    t3 = M[2, 3]

    mat_intrin = np.array([
            [fx,  0, cx, 0],
            [ 0, fy, cy, 0], 
            [ 0,  0,  1, 0],
            [ 0,  0,  0, 1]
        ])
    mat_extrin = np.eye(4)
    mat_extrin[0, :3] = r1
    mat_extrin[1, :3] = r2
    mat_extrin[2, :3] = r3
    mat_extrin[0,  3] = t1
    mat_extrin[1,  3] = t2
    mat_extrin[2,  3] = t3

    return [mat_intrin, mat_extrin]

def decompose_projection_mat_by_rq(mat_projection: np.ndarray):
    """
        RQ分解投影矩阵,旋转矩阵正交, 但内参skew因子不一定为0
    """
    M = mat_projection

    mat_intrin = np.eye(4)
    mat_extrin = np.eye(4)

    I = np.eye(3)
    P = np.flip(I, 1)
    A = M[:3, :3]
    _A = P @ A
    _Q, _R = np.linalg.qr(_A.T)
    R = P @ _R.T @ P
    Q = P @ _Q.T
    # check
    # print(R @ Q - A < 1E-10)
    
    mat_intrin[:3, :3] = R 
    mat_extrin[:3, :3] = Q 
    mat_extrin[:3,  3] = np.linalg.inv(R) @ M[:3, 3]
    return [mat_intrin, mat_extrin]

def project_points3d_to_2d(rtvec: np.ndarray, mat_projection: np.ndarray, points3d: np.ndarray)-> np.ndarray:
    """
        将3d点投影至2d
    """
    points3d = np.array(points3d)
    P = np.hstack((points3d, np.ones((points3d.shape[0], 1)))).T
    M = mat_projection

    rvec = rtvec[:3]
    tvec = rtvec[3:]
    #rvec[0] = 0
    #rvec[2] = 0

    T = t_to_T(tvec)
    R = r_to_R(rvec)

    V = T @ R

    points3d_ = (M @ V @ P)
    #points3d_ = points3d_ / points3d_[2]
    points2d = points3d_[:2, :] / points3d_[2]
    points2d = points2d.T
    return points2d

def rotation2d(theta):
    R = np.eye(3)
    R[:2, :2] = np.array([
        [ np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    return R

