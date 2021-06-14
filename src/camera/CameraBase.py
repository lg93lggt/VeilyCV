
import copy
from pathlib import Path
from src.visualizer import to_plot
from scipy.spatial.transform import Rotation
import cv2
import numpy as np
from numpy  import typing as npt
import open3d as o3d
import pandas as pd
from easydict import EasyDict
from icecream import ic
from src import geometries

try:
    import plugins
    from CommonGeometricShapes import Cuboid, GeometricShape
    from geometries import *
except:
    from src import plugins
    from src.CommonGeometricShapes import Cuboid, GeometricShape
    from src.geometries import *


class Intrinsic(object):
    def __init__(self, mat: np.ndarray=np.eye(4)):
        """---
        __init__ 
        ----------
            Initialize the intrinsic.
        
        Parameters
        ----------
        ### - `mat` : np.ndarray, optional, default=np.eye(4)
            intrinsic matrix, supported shapes:
            - (3, 3)
            - (4, 4)
        
        Returns
        -------
            None
        """        
        self.set_by_matrix(mat)
        return

    def info(self) -> None:
        """---
        info
        -------
            Print information of intrinsic.
        
        Returns
        -------
            None
        """        
        ic(self.__mat)
        return

    def set_by_params(self, fx: float, fy: float, cx: float, cy: float, skew: float=0):
        """---
        set_params 
        ----------
            Set intrinsic by parameters.
        
        Parameters
        ----------
        ### - `fx` : float
            
        ### - `fy` : float
            
        ### - `cx` : float
            
        ### - `cy` : float
            
        ### - `skew` : float, optional, default=0
            skew parameter of camera, usually equal to 0
        
        Returns
        -------
            None
        """        
        K: np.ndarray = np.diag(([fx, fy, 1., 1.]))
        K[:2, 2] = np.array([cx, cy])
        K[0, 1] = skew
        self.__mat = K
        return

    def set_by_matrix(self, mat: np.ndarray):
        """---
        set_matrix
        ----------
            Set intrinsic by matrix.
        
        Parameters
        ----------
        ### - `mat` : np.ndarray
            intrinsic matrix, supported shapes:
            - (3, 3)
            - (4, 4)
            
        Returns
        -------
            None
        
        Raises
        ------
        - `IndexError`
            [Shape of matrix is unsupported.]
        """                
        mat = np.asarray(mat)
        self.__mat: np.ndarray = np.eye(4)
        if mat.shape == (3, 3):
            self.__mat[:3, :3] = np.array(mat)
        elif mat.shape == (4, 4):
            self.__mat = np.array(mat)
        else:
            raise IndexError("Shape of matrix {} is unsupported.".format(mat.shape))
        return

    def _fx(self) -> float:
        """---
        _fx 
        -------
            Get fx.
        
        Returns
        -------
        ### - `fx`: float
            
        """
        fx = self.__mat[0, 0]
        return fx

    def _fy(self) -> float:
        """---
        _fy 
        -------
            Get fy.
        
        Returns
        -------
        ### - `fy`: float
            
        """
        fy = self.__mat[1, 1]
        return fy

    def _cx(self) -> float:
        """---
        _cx 
        -------
            Get cx.
        
        Returns
        -------
        ### - `cx`: float
            
        """
        return self.__mat[0, 2]

    def _cy(self) -> float:
        """---
        _cy 
        -------
            Get cy.
        
        Returns
        -------
        ### - `fy`: float
            
        """
        return self.__mat[1, 2]

    def _skew(self) -> float:
        """---
        _skew
        -------
            Get skew.
        
        Returns
        -------
        ### - `skew`: float
            
        """
        skew = self.__mat[0, 1]
        return skew

    def __mat_3x3(self) -> np.ndarray:
        """---
        _mat_3x3
        ----------
            Get 3x3 intrinsic matrix K.
        
        Parameters
        ----------
            None
            
        Returns
        -------
        ### - `K` : np.ndarray
            ouput intrinsic matrix, shape of K is (3, 3)
        
        Raises
        ------
            None
        """     
        K: np.ndarray = np.eye(3)
        K = self.__mat[:3, :3]
        return K

    def __mat_4x4(self) -> np.ndarray:
        """---
        _mat_4x4
        ----------
            Get 4x4 intrinsic matrix K.
        
        Parameters
        ----------
            None
            
        Returns
        -------
        ### - `K` : np.ndarray
            ouput intrinsic matrix, shape of K is (4, 4)
        
        Raises
        ------
            None
        """     
        K: np.ndarray = self.__mat
        return K

    def _mat(self, shape: Tuple[int]=(3, 3)) -> np.ndarray:
        """---
        _mat
        ----------
            Get intrinsic matrix K.
        
        Parameters
        ----------
        ### - `shape` : Tuple[int], optional, default=(3, 3)
            shape of ouput intrinsic matrix, supported sizes:
            - (3, 3)
            - (4, 4)
            
        Returns
        -------
        ### - `K` : np.ndarray
            ouput intrinsic matrix, shape of K is up to shape parameter
        
        Raises
        ------
        - `IndexError`
            [Shape of matrix is unsupported.]
        """      
        if   shape==(3, 3):
            K: np.ndarray = self.__mat_3x3()
        elif shape==(4, 4):
            K: np.ndarray = self.__mat_4x4()
        else:
            raise IndexError("Shape of matrix {} is unsupported.".format(shape))
        return K

    def _mat_inv(self, shape: Tuple[int]=(3, 3)) -> np.ndarray:
        """---
        _mat_inv
        ----------
        
        Get inverse intrinsic matrix KI.
        
        Parameters
        ----------
        ### - `shape` : Tuple[int], optional, default=(3, 3)
            shape of ouput inverse intrinsic matrix, supported sizes:
            - (3, 3)
            - (4, 4)
            
        Returns
        -------
        ### - `KI` : np.ndarray
            output inverse matrix of intrinsic matrix
        """        
        K: np.ndarray = self._mat(shape=shape)
        KI: np.ndarray = np.linalg.inv(K)
        return KI


class Extrinsic(object):
    def __init__(self, mat: np.ndarray=np.eye(4)):
        """---
        __init__ 
        ----------
            Initialize the extrinsic.
        
        Parameters
        ----------
        ### - `mat` : np.ndarray, optional, default=np.eye(4)
            extrinsic matrix, supported shapes:
            - (3, 4)
            - (4, 4)
        
        Returns
        -------
            None
        """        
        self.__R: Rotation   = Rotation()
        self.__t: np.ndarray = np.zeros((1, 3))
        self.set_by_matrix(mat)
        return

    def info(self) -> None:
        """---
        info
        -------
            Print information of intrinsic.
        
        Returns
        -------
            None
        """      
        ic(self.__mat)
        return

    def set_by_matrix(self, mat: np.ndarray):
        """---
        set_matrix
        ----------
            Set extrinsic by matrix.
        
        Parameters
        ----------
        ### - `mat` : np.ndarray
            extrinsic matrix, supported shapes:
            - (3, 4)
            - (4, 4)
            
        Returns
        -------
            None
        
        Raises
        ------
        - `IndexError`
            [Shape of matrix is unsupported.]
        """          
        assert mat[-1, -1] != 0
        mat = mat / mat[-1, -1]
        self.__mat: np.ndarray = np.eye(4)
        if mat.shape == (3, 4):
            self.__mat[:3] = np.array(mat)
        elif mat.shape == (4, 4):
            self.__mat = np.array(mat)
        else:
            raise IndexError("Shape of matrix {} is unsupported.".format(mat.shape))
        self.__R = Rotation.from_matrix(mat[:3, :3])
        self.__t = mat[:3, 3:4]
        return

    def set_by_rtvec(self, rvec: npt.ArrayLike[float]=np.zeros(3), tvec: npt.ArrayLike[float]=np.zeros(3)):
        """---
        set_by_rtvec 
        ----------
            Set extrinsic by rvec and tvec.
        
        Parameters
        ----------
        ### - `rvec` : npt.ArrayLike[float], optional, default=np.zeros(3)
            rotation vector
            
        ### - `tvec` : npt.ArrayLike[float], optional, default=np.zeros(3)
            translation vector 
        
        Returns
        -------
            None
        """        
        rvec: np.ndarray = np.asarray(rvec)
        tvec: np.ndarray = np.asarray(tvec)
        M = geometries.rtvec_to_transform_matrix(rvec=rvec, tvec=tvec, shape=(4, 4))
        self.set_by_matrix(M)
        return

    def set_by_Rmat_tvec(self, R: np.ndarray, tvec: npt.ArrayLike[float]):
        """---
        set_by_Rmat_tvec
        ----------
            Set extrinsic by 3x3 R matrix and t vector.
        
        Parameters
        ----------
        ### - `R` : np.ndarray
            (3, 3) rotation matrix
            
        ### - `tvec` : npt.ArrayLike[float]
            translation vector
        
        Returns
        -------
            None
        """        
        tvec: np.ndarray = np.asarray(tvec)
        tvec = tvec.flatten()
        M = np.eye(4)
        M[:3, :3] = R
        M[:3,  3] = tvec
        geometries.R_tvec
        self.set_by_matrix(M)
        return 

    def _rvec(self) -> np.ndarray:
        """---
        _rvec 
        -------
            Get rvec.
        
        Returns
        -------
        ### - `rvec`: np.ndarray
            output rotation vector
        """
        rvec: np.ndarray = self.__R.as_rotvec()
        return rvec
    
    def _tvec(self) -> np.ndarray:
        """---
        _tvec 
        -------
            Get tvec.
        
        Returns
        -------
        ### - `tvec`: np.ndarray
            output translation vector
        """
        tvec: np.ndarray = self.__mat[:3, 3]
        return tvec

    def _qaut(self) -> np.ndarray:
        """---
        _qaut 
        -------
            Get qaut.
        
        Returns
        -------
        ### - `qvec`: np.ndarray
            output quaternion rotation vector
        """
        qvec: np.ndarray = self.__R.as_quat()
        return qvec
    
    def __R_3x3(self) -> np.ndarray:
        """---
        __R_3x3 
        -------
            Get 3x3 rotation matrix.
        
        Returns
        -------
        ### - `R`: np.ndarray
            output (3, 3) rotation matrix
        """
        R: np.ndarray = self.__R.as_matrix()
        return R

    def __R_4x4(self) -> np.ndarray:
        """---
        _R_4x4 
        -------
            Get 4x4 rotation matrix.
        
        Returns
        -------
        ### - `R`: np.ndarray
            output (4, 4) rotation matrix
        """
        R: np.ndarray = np.eye(4)
        R[:3, :3] = self.__mat[:3, :3]
        return R

    def _R(self, shape: Tuple[int]=(4, 4)) -> np.ndarray:
        """---
        _R
        ----------
            Get rotation matrix.
        
        Parameters
        ----------
        ### - `shape` : Tuple[int], optional, default=(3, 3)
            shape of ouput rotation matrix, supported sizes:
            - (3, 3)
            - (4, 4)
            
        Returns
        -------
        ### - `R` : np.ndarray
            ouput rotation matrix, shape of R is up to shape parameter.
        
        Raises
        ------
        - `IndexError`
            [Shape is unsupported.]
        """      
        if   shape==(3, 3):
            R: np.ndarray = self.__R_3x3()
        elif shape==(4, 4):
            R: np.ndarray = self.__R_4x4()
        else:
            raise IndexError("Shape {} is unsupported.".format(shape))
        return R
    
    def _T(self) -> np.ndarray:
        """---
        _T 
        -------
            Get 4x4 translation matrix.
        
        Returns
        -------
        ### - `T`: np.ndarray
            output (4, 4) translation matrix
        """
        T: np.ndarray = np.eye(4)
        T[:3, 3:4] = self.__t
        return T

    def _R_inv(self, shape: Tuple[int]=(4, 4)) -> np.ndarray:
        """---
        _R_inv
        ----------
            Get inverse rotation matrix RI.
        
        Parameters
        ----------
        ### - `shape` : Tuple[int], optional, default=(3, 3)
            shape of ouput inverse rotation matrix, supported sizes:
            - (3, 3)
            - (4, 4)
            
        Returns
        -------
        ### - `RI` : np.ndarray
            ouput inverse rotation matrix, shape of RI is up to shape parameter.
        
        Raises
        ------
        - `IndexError`
            [Shape of matrix is unsupported.]
        """      
        if   shape==(3, 3):
            R: np.ndarray = self.__R_3x3()
        elif shape==(4, 4):
            R: np.ndarray = self.__R_4x4()
        else:
            raise IndexError("Shape of matrix {} is unsupported.".format(shape))
        RI: np.ndarray = R.T
        return RI

    def _T_inv(self) -> np.ndarray:
        """---
        _T_inv
        ----------
            Get inverse translation matrix TI.
        
        Parameters
        ----------
            None
            
        Returns
        -------
        ### - `TI` : np.ndarray
            ouput inverse translation matrix, shape of TI is (4, 4).
        
        Raises
        ------
            None
        """      
        TI: np.ndarray = np.eye(4)
        TI[:3, 3:4] = -self.__t
        return TI
    
    def __mat_3x4(self) -> np.ndarray:
        """---
        __mat_3x4
        -------
            Get 3x4 extrinsic matrix.
        
        Returns
        -------
        ### - `M`: np.ndarray
            output (3, 4) translation matrix
        """        
        M: np.ndarray = np.eye(3, 4)
        M = self.__mat[:3]
        return M

    def __mat_4x4(self) -> np.ndarray:
        """---
        __mat_4x4
        -------
            Get 4x4 extrinsic matrix.
        
        Returns
        -------
        ### - `M`: np.ndarray
            output (4, 4) translation matrix
        """        
        M: np.ndarray = self.__mat.copy()
        return M

    def _mat(self, shape: Tuple[int]=(4, 4)) -> np.ndarray:
        """---
        _mat
        ----------
            Get extrinsic matrix M.
        
        Parameters
        ----------
        ### - `shape` : Tuple[int], optional, default=(4, 4)
            shape of ouput extrinsic matrix, supported sizes:
            - (3, 4)
            - (4, 4)
            
        Returns
        -------
        ### - `M` : np.ndarray
            ouput extrinsic matrix, shape of M is up to shape parameter.
        
        Raises
        ------
        - `IndexError`
            [Shape of matrix is unsupported.]
        """      
        if   shape==(3, 3):
            M: np.ndarray = self.__mat_3x4()
        elif shape==(4, 4):
            M: np.ndarray = self.__mat_4x4()
        else:
            raise IndexError("Shape of matrix {} is unsupported.".format(shape))
        return M

    def _cvec(self) -> np.ndarray:
        """---
        _cvec 
        -------
            Get cvec.
        
        Returns
        -------
        ### - `cvec`: np.ndarray
            output cemera center vector in world coordinate
        """
        cvec:  np.ndarray = - self._R_inv(shape=(3, 3)) @ self.__t
        cvec = cvec.flatten()
        return cvec

    def _mat_inv(self, shape: Tuple[int]=(4, 4)) -> np.ndarray:
        """---
        _mat_inv
        ----------
            Get inverse extrinsic matrix MI.
        
        Parameters
        ----------
        ### - `shape` : Tuple[int], optional, default=(4, 4)
            shape of ouput extrinsic matrix, supported sizes:
            - (3, 4)
            - (4, 4)
            
        Returns
        -------
        ### - `MI` : np.ndarray
            ouput inverse extrinsic matrix, shape of M is up to shape parameter.
        
        Raises
        ------
        - `IndexError`
            [Shape of matrix is unsupported.]
        """      
        MI: np.ndarray = self._R_inv(shape=(4, 4)) @ self._T_inv()
        if   shape==(3, 4):
            MI = MI[:3]
        elif shape==(4, 4):
            pass
        else:
            raise IndexError("Shape of matrix {} is unsupported.".format(shape))
        return MI

        
class CameraBase(object):
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.height: int = height
        self.width : int = width
        self.__update_image_size()
        return

    def __update_image_size(self):
        self.size_image: Tuple = (self.height, self.width)
        return

class IdealCamera(GeometricShape.GeometricShape):
    def __init__(self, height, width, fov_degree=60):
        super().__init__()

        self.fov_rad = fov_degree/180*np.pi
        K = plugins.params_o3d_to_intrinsic_cv(
            self.fov_rad, width=width, height=height)

        self.height = height
        self.width = width
        self.size_image = (height, width)
        self.intrinsic = Intrinsic(K)
        self.extrinsic = Extrinsic()

        self.visualize_default_settings()
        return

    def info(self) -> None:
        ic(self.intrinsic.__mat_3x3())
        ic(self.extrinsic._mat_3x4())
        ic(self.size_image)
        return

    def _projection_matrix_4x4(self):
        P = self.intrinsic.__mat_4x4() @ self.extrinsic._mat_4x4()
        return P

    def _projection_matrix_3x4(self):
        P = self.intrinsic._mat() @ self.extrinsic._mat()
        return P

    def set_projection_matrix(self, P):
        [K, M] = decompose_projection_mat(P)
        self.intrinsic.set_by_matrix(K)
        self.extrinsic.set_by_matrix(M)
        return

    def set_extrinsic(self, extrinsic):
        self.extrinsic = extrinsic
        return

    def set_extrinsic_matrix(self, M):
        self.extrinsic.set_by_matrix(M)
        return

    def set_extrinsic_matrix_identity(self):
        self.extrinsic.set_by_matrix(np.eye(4))
        return

    def set_extrinsic_by_rtvec(self, rvec=np.zeros(3), tvec=np.zeros(3)):
        M = rtvec_to_transform_matrix(rvec=rvec, tvec=tvec, shape=(4, 4))
        self.extrinsic.set_by_matrix(M)
        return

    def set_extrinsic_by_rcvec(self, rvec=np.zeros(3), cvec=np.zeros(3)):
        R = geometries.rvec_to_Rmat(rvec)
        C = geometries.t_to_T(cvec)
        M = R @ C
        self.extrinsic.set_by_matrix(M)
        return

    def set_lookat(self, front, up, at, zoom):
        [self.at, self.eye, self.up] = plugins.params_o3d_to_params_opengl(
            self.fov_rad, front, up, at, zoom)
        M = plugins.params_o3d_to_extrinsic_cv(
            fov_rad=self.fov_rad, front=front, up=up, at=at, zoom=zoom)
        self.extrinsic.set_by_matrix(M)
        return

    def get_optical_center_3d(self):
        return self.extrinsic._c()

    def visualize_default_settings(self):
        self.render_options = EasyDict({})

        self.render_options.coord = EasyDict({})
        self.render_options.coord.show = True
        self.render_options.coord.length = 0.1

        self.render_options.optical_center = EasyDict({})
        self.render_options.optical_center.show = 0
        self.render_options.optical_center.radius = 0.01

        self.render_options.optical_axis = EasyDict({})
        self.render_options.optical_axis.show = False
        self.render_options.optical_axis.expand = False 
        self.render_options.optical_axis.scale = 1
        #self.render_options.optical_axis.colors = False
        self.render_options.optical_axis.uniform_color = np.full(3, 255)

        self.render_options.frustum = EasyDict({})
        self.render_options.frustum.show = False
        self.render_options.frustum.expand = False
        return

    def render_coord(self, vis):
        if self.render_options.coord.show:
            self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
                self.render_options.coord.length)
            self.coord = self.coord.transform(
                np.linalg.inv(self.extrinsic._mat_4x4()))
            vis.add_geometry(self.coord)
        else:
            return

    def render_optical_center(self, vis):
        if self.render_options.optical_center.show:
            self.optical_center = o3d.geometry.TriangleMesh.create_sphere(
                self.render_options.optical_center.radius)
            self.optical_center = self.optical_center.translate(
                self.extrinsic._c())
            vis.add_geometry(self.optical_center)
        else:
            return

    def render_optical_axis(self, vis):
        if self.render_options.optical_axis.show:
            vertexes = np.array([[0, 0, 0], [0, 0, 1]], dtype=float)
            lines = np.array([[0, 1]], dtype=int)

            if self.render_options.optical_axis.expand:
                s = self.render_options.optical_axis.scale
            else:
                s = 1
            self.optical_axis = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(vertexes * s),
                lines=o3d.utility.Vector2iVector(lines)
            )
            self.optical_axis.paint_uniform_color([1, 1, 1])
            self.optical_axis = self.optical_axis.transform(
                np.linalg.inv(self.extrinsic._mat_4x4()))
            vis.add_geometry(self.optical_axis)
        else:
            return

    def render_axis(self, vis):
        if self.render_options.axis.show:
            self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                self.render_options.axis.length)
            self.axis = self.axis.transform(
                np.linalg.inv(self.extrinsic._mat_4x4()))
            vis.add_geometry(self.axis)
        else:
            return

    def view_at_camera_axes(self, vis, block=True, render_axis=False):
        ctl = vis.get_view_control()
        ctl.set_constant_z_far(100)
        if render_axis:
            self.render_coord(vis)
            self.render_optical_axis(vis)
            self.render_optical_center(vis)

        params = o3d.camera.PinholeCameraParameters()
        intrin = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.intrinsic._fx(), self.intrinsic._fy(), self.intrinsic._cx(), self.intrinsic._cy())
        params.intrinsic = intrin
        params.extrinsic = self.extrinsic._mat_4x4()
        ctl.convert_from_pinhole_camera_parameters(params, True)

        vis.update_renderer()
        vis.poll_events()
        if block:
            vis.run()
        return

    def view_at_image_axes(self, vis):
        self.view_at_camera_axes(vis)
        return

    def project_points(self, points3d):
        points2d = project_points3d_to_2d(rtvec=np.zeros(6), mat_projection=self._projection_matrix_4x4(), points3d=points3d)
        return points2d

    def project_points_on_image(self, points3d, color=(255, 255, 255), radius=1, thickness=1, img=None) -> np.ndarray:
        if img is None:
            img = np.zeros((self.height, self.width, 3)).astype(np.uint8)
        else:
            pass
        points2d = self.project_points(points3d)
        for pt in points2d:
            pt = np.round(pt).astype(np.int)
            img = cv2.circle(img=img, center=tuple(pt), radius=radius, color=color, thickness=thickness)
        return img

    def project_grid_on_image(self, grid, color=(255, 255, 255), radius=1, thickness=1, img=None) -> np.ndarray:
        if img is None:
            img = np.zeros((self.height, self.width, 3)).astype(np.uint8)
        else:
            pass
        m = grid.reshape((3, -1)).max(axis=1)
        n = grid.reshape((3, -1)).min(axis=1)
        d = m - n
        points3d = np.array([
            n,
            n + np.array([d[0],    0,    0]),
            n + np.array([d[0], d[1],    0]),
            n + np.array([   0, d[1],    0]),

            n + np.array([   0,    0, d[2]]),
            n + np.array([d[0],    0, d[2]]),
            m,
            n + np.array([   0, d[1], d[2]]),
        ])

        pts = self.project_points(points3d)
        pts = np.round(pts).astype(int)
        for i in range(4):
            j = i + 4
            if i<3:
                img = cv2.line(img=img, pt1=tuple(pts[i]), pt2=tuple(pts[i + 1]), color=color, thickness=thickness)
                img = cv2.line(img=img, pt1=tuple(pts[j]), pt2=tuple(pts[j + 1]), color=color, thickness=thickness)
            elif i == 3:
                img = cv2.line(img=img, pt1=tuple(pts[i]), pt2=tuple(pts[0]), color=color, thickness=thickness)
                img = cv2.line(img=img, pt1=tuple(pts[j]), pt2=tuple(pts[4]), color=color, thickness=thickness)
            else:
                break
            img = cv2.line(img=img, pt1=tuple(pts[i]), pt2=tuple(pts[j    ]), color=color, thickness=thickness)
        return img

    def project_axis_on_image(self, unit_length=0.1, width_line=1, img=None) -> np.ndarray:
        if img is None:
            img = np.zeros((self.height, self.width, 3)).astype(np.uint8)
        
        points3d = np.zeros((4, 3))
        if   isinstance(unit_length, np.ndarray):
            if len(unit_length)==3:
                points3d[1:, :] = np.diag(unit_length)
            else:
                raise IndexError("unit_length should be a 1*3 vector or a float.")
        elif isinstance(unit_length, float):
            points3d = np.array([[0, 0, 0], [unit_length, 0, 0], [0, unit_length, 0], [0, 0, unit_length]])
        else:
            raise IndexError("unit_length should be a 1*3 vector or a float.")
        p2ds = self.project_points(points3d=points3d)
        try:
            cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[1]), (0, 0, 255), width_line)
            cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[2]), (0, 255, 0), width_line)
            cv2.line(img, to_plot(p2ds[0]), to_plot(p2ds[3]), (255, 0, 0), width_line)
        except :
            pass
        return img

    def copy(self):
        return copy.deepcopy(self)

    def resize(self, **kwargs: Tuple[Tuple[int], float]) -> IdealCamera:
        """---
        # resize
        change camera intrinsic according to resize image

        Parameters
        -------
        ### - `shape`:
            new image shape  
        ### - `scale`: 
            get new image shape by scale, (h2, w2) = (h1, h1) * scale
            
        Returns
        -------
        [type]
            [description]
        """        
        camera_new = self.copy()
        K = camera_new.intrinsic.__mat_4x4()
        if "scale" in kwargs.keys():
            scale = kwargs["scale"]
            s = 1 / scale 
            if not (camera_new.height % s) == 0 and (camera_new.width % s == 0):
                Warning("s value is unsupported.")
                return
            else:
                camera_new.height = int(camera_new.height / s)
                camera_new.width  = int(camera_new.width  / s)
                camera_new.size_image = (camera_new.height, camera_new.width)
        elif "shape" in kwargs.keys():
            size_new = kwargs["shape"]
            if  (camera_new.height / size_new[0] == camera_new.height / size_new[0]) \
                and (camera_new.width  % size_new[1] == 0) \
                and (camera_new.height % size_new[0] == 0):
                    s = camera_new.height / size_new[0]
                    camera_new.height = int(camera_new.height / s)
                    camera_new.width  = int(camera_new.width  / s)
                    camera_new.size_image = (camera_new.height, camera_new.width)
            else:
                Warning("new shape is unsupported.")
                return

        K[0, 0] = K[0, 0] / s
        K[1, 1] = K[1, 1] / s
        K[0, 2] = K[0, 2] / s - 0.5 / s + 0.5
        K[1, 2] = K[1, 2] / s - 0.5 / s + 0.5
        K[2, 2] = 1
        K[3, 3] = 1
        camera_new.intrinsic.set_by_matrix(K)
        return camera_new

    def get_image_from_camera_axes(self, vis):
        self.view_at_camera_axes(vis, block=False)
        img_o3d = np.array(vis.capture_screen_float_buffer(False)) * 255.0
        img = img_o3d.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def get_image_from_image_axes(self, vis):
        self.view_at_image_axes(vis)
        img_o3d = np.array(vis.capture_screen_float_buffer(False)) * 255.0
        img = img_o3d.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def load_from_file(self, pth_input: Union[Path, str]):
        pth_input = Path(pth_input) if isinstance(pth_input, str) else pth_input
        if pth_input.suffix == ".pkl":
            data_load = pd.read_pickle(pth_input)
            data_load = data_load[data_load.notna()]
        else:
            raise NameError("Unsupported format {}.".format(pth_input.suffix))
        if   isinstance(data_load, pd.Series):
            if "intrinsic" in data_load.index:
                self.intrinsic.set_by_matrix(data_load.intrinsic.astype(float))
            if ("rvec" in data_load.index) and ("tvec" in data_load.index):
                rvec = data_load.height.astype(int)[0]
                tvec = data_load.height.astype(int)[0]
                self.set_extrinsic_by_rtvec(rvec=rvec, tvec=tvec)
            if "height" in data_load.index:
                self.height = data_load.height.astype(int)[0]
            if "width" in data_load.index:
                self.width  = data_load.width.astype(int)[0]
            self.size_image = (self.height, self.width)
        elif isinstance(data_load, pd.DataFrame):
            if ("rvec" in data_load.columns) and ("tvec" in data_load.columns):
                rvecs = np.asarray(data_load.rvec.to_list(), dtype=float)
                tvecs = np.asarray(data_load.tvec.to_list(), dtype=float)
                self.trajectory = geometries.rtvecs_to_transform_matrices(rvecs=rvecs, tvecs=tvecs)
                self.names_traj = data_load.index.tolist()
        return 

    def apply_trajectory_by_index(self, idx: int) -> str:
        try:
            traj = self.trajectory
            self.set_extrinsic_matrix(traj[idx])
            return self.names_traj[idx]
        except :
            print("No traj.")
            return


class PinholeCamera(IdealCamera):
    def __init__(self, height, width, K=np.eye(3)) -> None:
        super().__init__(height=height, width=width)
        if (K == np.eye(3)).all():
            K = np.array([
                [415.69219382, 0, height/2 - 0.5],
                [0, 415.69219382, width /2 - 0.5],
                [0, 0, 1],
            ])
        self.intrinsic = Intrinsic(K)
        return

    def view_at_image_axes(self, vis):
        ctl = vis.get_view_control()
        ctl.set_constant_z_far(100)
        self.render_coord(vis)
        self.render_optical_center(vis)
        self.render_optical_axis(vis)

        params = o3d.camera.PinholeCameraParameters()
        intrin = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.intrinsic._fx(), self.intrinsic._fy(), self.intrinsic._cx(), self.intrinsic._cy())
        params.intrinsic = intrin
        params.extrinsic = self.extrinsic._mat_4x4()
        ctl.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

        fov = ctl.get_field_of_view()
        cx_ideal = self.width  / 2 - 0.5
        cy_ideal = self.height / 2 - 0.5

        delta_cx = self.intrinsic._cx() - cx_ideal
        delta_cy = self.intrinsic._cy() - cy_ideal

        ctl.translate(
            delta_cx/self.intrinsic._fx()*self.width,
            delta_cy/self.intrinsic._fy()*self.height
        )
        vis.update_renderer()
        vis.poll_events()

        return



if __name__ == '__main__':
    [H, W] = [1440, 1080, ]  # [1280, 1920]#

    cam1 = IdealCamera(height=H, width=W)
    cam1.render_options.coord.show = True
    cam1.render_options.coord.length = 0.1
    cam1.set_lookat(front=np.array([1, 1, 0.]), up=np.array([0, 1, 0.]), at=np.array([1, 1, 1.]), zoom=1)

    cam2 = IdealCamera(height=H, width=W)
    cam2.set_lookat(front=np.array([1, 1, 0.]), up=np.array([0, 1, 0.]), at=np.array([1.5, 1.5, 1.5]), zoom=1)

    vis = o3d.visualization.Visualizer()
    vis.create_window("o3d", width=W, height=H, visible=1)


    cube = Cuboid.Cuboid(0.1, 0.1, 0.1)
    cube.draw(vis)
    
    img = cam1.get_image_from_image_axes(vis)

    cam1.view_at_camera_axes(vis, render_axis=1)
    cam2.view_at_camera_axes(vis, render_axis=1)
    vis.run()
    #cam2.view_at_camera_axes(vis, block=True, render_axis=1)
    
    

        