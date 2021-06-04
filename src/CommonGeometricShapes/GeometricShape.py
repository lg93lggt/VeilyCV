
import copy
import numpy as np
import open3d as o3d
from typing import *
from easydict import EasyDict

try:
    from .. import geometries
except :
    from src import geometries


class GeometricShape(object):

    def __init__(self):
        self.vertexes = np.ones((1, 3))

        self.mesh    = o3d.geometry.TriangleMesh()
        self.mesh.vertices  = o3d.utility.Vector3dVector(self.vertexes)

        self.lineset = o3d.geometry.LineSet()
        self.lineset.points = o3d.utility.Vector3dVector(self.vertexes)
        
        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
        return
        
    def _mesh(self):
        mesh          = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertexes)
        return mesh

    def _lineset(self):
        lineset          = o3d.geometry.LineSet()
        lineset.vertices = o3d.utility.Vector3dVector(self.vertexes)
        return lineset

    def set_uniform_color(self, color: np.ndarray=np.array([255, 255, 255]), object: str="mesh"):
        self.colors = EasyDict({})
        if   object == "mesh":
            self.colors.mesh    = np.ones_like(self.vertexes) * np.array(color) / 255
            self.mesh.colors    = o3d.utility.Vector3dVector(self.colors.mesh)
        elif object == "lineset":
            self.colors.lineset = np.ones((len(self.lineset.lines), 3)) * np.array(color) / 255
            self.lineset.colors = o3d.utility.Vector3dVector(self.colors.lineset)
        return self

    def transform3d(self, M: np.ndarray=np.eye(4)):
        P = geometries.transform3d(M, self.vertexes, axis=0, is_homo=False)
        self.vertexes = P[:3].T
        self.mesh.vertices  = o3d.utility.Vector3dVector(self.vertexes)
        self.lineset.points = o3d.utility.Vector3dVector(self.vertexes)
        self.axes.transform(M)
        return self

    def transform_by_rtvecs(self, rvec: np.ndarray, tvec: np.ndarray):
        M = geometries.rtvec_to_transform_matrix(rvec=rvec, tvec=tvec)
        P = geometries.transform3d(M, self.vertexes, axis=0, is_homo=False)
        self.vertexes = P[:3].T
        self.mesh.vertices  = o3d.utility.Vector3dVector(self.vertexes)
        self.lineset.points = o3d.utility.Vector3dVector(self.vertexes)
        self.axes.transform(M)
        return self

    def translate3d(self, t: np.ndarray=np.zeros(3)):
        T = geometries.t_to_T(t)
        P = geometries.transform3d(T, self.vertexes, axis=0, is_homo=False)
        self.vertexes = P[:3].T
        self.mesh.vertices  = o3d.utility.Vector3dVector(self.vertexes)
        self.lineset.points = o3d.utility.Vector3dVector(self.vertexes)
        self.axes.transform(T)
        return self

    def self_translate3d(self, t: np.ndarray=np.zeros(3)):
        T = geometries.t_to_T(t)
        P = geometries.transform3d(T, self.vertexes, axis=0, is_homo=False)
        self.vertexes = P[:3].T
        self.mesh.vertices  = o3d.utility.Vector3dVector(self.vertexes)
        self.lineset.points = o3d.utility.Vector3dVector(self.vertexes)
        self.axes.transform(T)
        return self

    def draw_mesh(self, vis: o3d.visualization.Visualizer):
        try:
            vis.add_geometry(self.mesh)
        except :
            pass
        return

    def draw_axes(self, vis: o3d.visualization.Visualizer):
        try:
            vis.add_geometry(self.axes)
        except :
            pass
        return

    def draw_lineset(self, vis: o3d.visualization.Visualizer):
        try:
            vis.add_geometry(self.lineset)
        except :
            pass
        return

    def draw(self, vis: o3d.visualization.Visualizer):
        self.draw_mesh(vis)    
        self.draw_lineset(vis) 
        self.draw_axes(vis)
        return  
        
    def delete_mesh(self, vis: o3d.visualization.Visualizer):
        try:
            vis.remove_geometry(self.mesh)
        except :
            pass
        return

    def delete_axes(self, vis: o3d.visualization.Visualizer):
        try:
            vis.remove_geometry(self.axes)
        except :
            pass
        return
        
    def delete_lineset(self, vis: o3d.visualization.Visualizer):
        try:
            vis.remove_geometry(self.lineset)
        except :
            pass
        return


    def delete(self, vis: o3d.visualization.Visualizer):
        self.delete_mesh(vis)
        self.delete_lineset(vis)
        self.delete_axes(vis)
        return
    
    def copy(self):
        return copy.deepcopy(self)
