
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import OpenGL.arrays.vbo
from typing import Union, Tuple
import open3d as o3d
import numpy as np

from src.vl3d.structures.MeshBase import MeshBase
from src.vl3d.structures.Axis import Axis

class RendererOpenGL(object):
    def __init__(self, **kwargs):
        self.__update_image_size(**kwargs)
        ax = Axis()
        self.scene = [ax]
        return
    
    def __update_image_size(self, **kwargs: Union[Tuple[int], int]):
        keys = kwargs.keys()
        key_height     = "height"
        key_width      = "width"
        key_size_window = "size_window"
        if (key_height in keys) and (key_width in keys):
            self.height = kwargs[key_height]
            self.width  = kwargs[key_width]
            self.size_image = (self.height, self.width)
        elif key_size_window in keys:
            self.size_window = kwargs[key_size_window]
            self.height = self.size_image[0]
            self.width  = self.size_image[1]
        else:
            raise KeyError("Using unsupported keys.")
        return
    
    def __init_opengl(self):
        """---
        # __init_opengl
        initial renderer
        """        
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)
        glDepthFunc(GL_ALWAYS)
        
        return
    
    # def draw_mesh(self, mesh: o3d.geometry.TriangleMesh):
    #     verts = np.asarray(mesh.vertices).astype(np.float32)
    #     faces = np.asarray(mesh.triangles).astype(np.uint32)
    #     self.vbo = OpenGL.arrays.vbo.VBO(data=np.hstack((np.ones_like(verts, dtype=np.float32), verts, )), target=GL_ARRAY_BUFFER)
    #     self.ebo = OpenGL.arrays.vbo.VBO(data=faces.flatten(), target=GL_ELEMENT_ARRAY_BUFFER)
        
    #     self.vbo.bind()
    #     glInterleavedArrays(GL_C3F_V3F, 0, None)
    #     self.ebo.bind()
      
    #     glPushMatrix()
    #     glDrawElements(GL_TRIANGLES, len(faces), GL_UNSIGNED_INT, None)
    #     glPopMatrix()
    #     return
    
    def append_scene(self, mesh: MeshBase):
        self.scene.append(mesh)
        return        
    
    def __draw_scene(self):
        """---
        # __draw_scene
        draw meshes in scene
        """        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # glPushMatrix()
        # glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        for mesh in self.scene:
            mesh._draw()
            mesh.draw()
        glutSwapBuffers()
        return
        
        
    def run(self):
        glutInit(sys.argv)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0 ,0)
        window = glutCreateWindow("test")
        glutDisplayFunc(self.__draw_scene)
        glutIdleFunc(self.__draw_scene)
        glutMouseFunc(None)
        glutKeyboardFunc(None)
        self.__init_opengl()
        glutMainLoop()
        
if __name__ == '__main__':
    ax = Axis()
    ren = RendererOpenGL(height=480, width=640)
    # mesh = o3d.io.read_triangle_mesh("/media/veily3/data_ligan/匹克脚部数据集/raw/000001/left.ply")
    # ren.append_scene(mesh)
    ren.run()
        