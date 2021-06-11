

import numpy as np
from OpenGL.GL import *
from OpenGL.arrays import vbo
from OpenGL.GLU import *
from OpenGL.GLUT import *


class MeshBase(object):
    def __init__(self, vertexes: np.ndarray, indexes: np.ndarray):
        self.vertexes = vertexes
        self.indexes  = indexes
        self.n_points = len(indexes)
        self.colors   = np.ones_like(vertexes, dtype=np.float32)
        self._update_vbo()
        self.ebo = vbo.VBO(data=indexes, usage=GL_DYNAMIC_DRAW, target=GL_ELEMENT_ARRAY_BUFFER)
        return
    
    
    def _update_vbo(self):
        verts = np.hstack((self.colors, self.vertexes))
        self.vbo = vbo.VBO(data=verts, usage="GL_DYNAMIC_DRAW", target="GL_ARRAY_BUFFER", size=None)
        return
        
    
    def _draw(self):
        self.vbo.bind()
        glInterleavedArrays(GL_C3F_V3F, 0, None)
        self.ebo.bind()

        #glPushMatrix()
        glLoadIdentity()
        # TODO glMultMatrixf(self.pose.flatten())
        glDrawElements(mode=GL_TRIANGLES, count=self.n_points, type=GL_UNSIGNED_INT, indices=self.ebo)
        #glPopMatrix()