
import numpy as np
import open3d as o3d
from typing import *
from easydict import EasyDict
import sys

try:
    from ..geometries import *
    import GeometricShape
except:
    from src.geometries import *
    from src.CommonGeometricShapes import GeometricShape


class Cuboid(GeometricShape.GeometricShape):
    """
    Define a cuboid shape: [Width, Height, Length] -> coord: [X, Y, X]
    """
    def __init__(self, width: float or int, height: float or int, length: float or int, center: np.ndarray=np.zeros(3)):
        """
        @ width : -> coord X
        @ height: -> coord Y
        @ length: -> coord Z
            shape = [width, height, length]
            verts:
                array([
                    [0, 0, 0],
                    [0, 0, L],
                    [0, H, 0],
                    [0, H, L],
                    [W, 0, 0],
                    [W, 0, L],
                    [W, H, 0],
                    [W, H, L]], dtype=np.float
                ) - (shape/2 - center)
        @ colors:
            array([
                [1, 1, 1],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]], dtype=np.float)
        """
        self.center=center
        self.shape = np.array([width, height, length])
        self.vertexes = np.mgrid[0:2, 0:2, 0:2].reshape((3, -1)).T * self.shape - self.shape / 2 + self.center
        self.faces = np.array([
            [0, 2, 1],
            [1, 2, 3], 
            [1, 3, 7], 
            [7, 5, 1], 
            [0, 1, 5], 
            [5, 4, 0], 
            [0, 4, 2], 
            [4, 6, 2], 
            [4, 5, 7], 
            [7, 6, 4], 
            [6, 7, 3], 
            [3, 2, 6]
        ], dtype=int)
        self.edges = np.array([
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7]
        ], dtype=int)

        self._refresh()
        #self.axes.translate(-center)
        return

    def _refresh(self):
        self.mesh               = o3d.geometry.TriangleMesh()
        self.mesh.vertices      = o3d.utility.Vector3dVector(self.vertexes)
        self.mesh.triangles     = o3d.utility.Vector3iVector(self.faces)
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones_like(self.vertexes))

        self.lineset            = o3d.geometry.LineSet()
        self.lineset.lines      = o3d.utility.Vector2iVector(self.edges)
        self.lineset.colors     = o3d.utility.Vector3dVector(np.zeros((12, 3)))
        self.lineset.points     = o3d.utility.Vector3dVector(self.vertexes)

        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(np.average(self.shape) / 2)
        return

    def draw_without_axis(self, vis):
        super().draw_mesh(vis)
        super().draw_lineset(vis)
        return


class ArucoCube(Cuboid):
    def __init__(self, length=1, center=np.zeros(3)):
        super().__init__(length, length, length, center)
        self.vertexes -= np.array([0,0, 0.12])
        self._refresh()
        return


if __name__ == '__main__':
    vis = o3d.visualization.Visualizer()
    vis.create_window("o3d", width=480, height=640)

    opt = vis.get_render_option()
    opt.background_color = np.zeros(3)


    cube = Cuboid(0.12, 0.12, 0.12, center=np.array([0, 0, 0.06]))

    #cube.translate3d(np.array([[0.1, 0.1, 0.1]]))
    #cube.transform3d(e1)
    cube.draw(vis)
    vis.run()

