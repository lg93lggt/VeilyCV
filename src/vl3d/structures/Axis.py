

import numpy as np
from OpenGL.GL import *
from OpenGL.arrays import vbo
from OpenGL.GLU import *
from OpenGL.GLUT import *

from src.vl3d.structures.MeshBase import MeshBase

class Axis(MeshBase):
    def __init__(self, lens=[1, 1, 1]):
        vertexes = np.eye(4, 3, k=-1, dtype=np.float32) * lens
        indexes  = np.array([[0, 1, 1], [0, 2, 2], [0, 3, 3]], dtype=np.uint32)
        super().__init__(vertexes, indexes)
        self.colors = np.array(
            [[1, 1, 1], 
             [1, 1, 1], 
             [1, 1, 1],
             [1, 1, 1]], 
            dtype=np.float32
        )
        #self._update_vbo()
        return

    def _update_vbo(self):
        super()._update_vbo()
        return
    
    def _draw(self):
        self.vbo.bind()
        glInterleavedArrays(GL_C3F_V3F, 0, None)
        self.ebo.bind()

        #glPushMatrix()
        glLoadIdentity()
        # TODO glMultMatrixf(self.pose.flatten())
        glDrawElements(GL_LINES, self.n_points, GL_UNSIGNED_INT, None)
        self.vbo.unbind()
        self.ebo.unbind()
        #glPopMatrix()
        
    def draw(self):
        
        glBegin(GL_LINES)
        for i_line in range(3):
            r = [1., 0., 0.][i_line]
            g = [0., 1., 0.][i_line]
            b = [0., 0., 1.][i_line]
            glColor3f(r, g, b)
            [idx1, idx2] = self.indexes[i_line]
            [x, y, z] = self.vertexes[idx1]
            glVertex3f(x, y, z)
            [x, y, z] = self.vertexes[idx2]
            glVertex3f(x, y, z)
        glEnd()