
import numpy as np
import open3d as o3d
from typing import *

from CommonGeometricShapes import GeometricShape

class ChessBoard(GeometricShape.GeometricShape):
    def __init__(self, pattern_size: Union[Tuple[int], List[int]],  unit_length: float=0.1):
        """
        """
        super().__init__()

        self.pattern_size = pattern_size

        self.n_units_in_h  = self.pattern_size[0] + 1
        self.n_units_in_w  = self.pattern_size[1] + 1

        self.n_pts_in_h  = self.pattern_size[0] + 2
        self.n_pts_in_w  = self.pattern_size[1] + 2

        self.vertexes = np.mgrid[0:self.n_pts_in_h + 1, 0:self.n_pts_in_w + 1, 0:1].T.reshape((-1, 3)) * unit_length
        #a = np.mgrid[0:self.pattern_size[0] + 2, 0:self.pattern_size[1] + 2].transpose((1, 2,0))
        l = []
        for i in range(self.n_units_in_h):
            for j in range(self.n_units_in_w):
                a = [   i  * (self.n_pts_in_w) +  j   , i * (self.n_pts_in_w) + j + 1, (i+1) * (self.n_pts_in_w) + j]
                b = [(i+1) * (self.n_pts_in_w) + (j+1), i * (self.n_pts_in_w) + j + 1, (i+1) * (self.n_pts_in_w) + j]
                l.append(a)
                l.append(b)
        l=np.array(l)
        colors_mesh       = np.zeros_like(self.vertexes)
        #colors_mesh[1::2] = np.ones(3)

        self.mesh.vertices  = o3d.utility.Vector3dVector(self.vertexes)
        self.mesh.triangles = o3d.utility.Vector3iVector(l)
        self.mesh.vertex_colors    = o3d.utility.Vector3dVector(colors_mesh)

        
        self.lineset.points = o3d.utility.Vector3dVector(self.vertexes)
        n_lines = self.n_pts_in_h * self.n_units_in_w + self.n_pts_in_w * self.n_units_in_h
        lines = np.zeros((n_lines, 2), dtype=int)
        #lines[:self.n_pts_in_h * self.n_units_in_w] = 
        lines[:self.n_pts_in_h * self.n_units_in_w] = np.fromfunction(lambda x, y: (x + y + x//self.n_units_in_w), (self.n_pts_in_h * self.n_units_in_w, 2))#.astype(int)
        lines[self.n_pts_in_h * self.n_units_in_w:] = np.fromfunction(lambda x, y: (x + y*self.n_pts_in_w), (self.n_pts_in_w * self.n_units_in_h, 2))#.astype(int)
        self.lineset.lines = o3d.utility.Vector2iVector(lines)
        return

    

if __name__ == '__main__':
    cb = ChessBoard([2, 3], 0.1)
    cb.set_uniform_color(np.array([255, 0, 0.]), "lineset")

    vis = o3d.visualization.Visualizer()
    vis.create_window("o3d", width=640, height=480)

    opt = vis.get_render_option()
    opt.background_color = np.zeros(3)
    cb.draw_lineset(vis)
    cb.draw_mesh(vis)
    vis.run()
    print()