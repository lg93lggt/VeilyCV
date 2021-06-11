
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import sys
import json
import numpy as np
import cv2
import sys
from copy import deepcopy
from PIL import Image
import time



dir_root = "/home/veily3/LIGAN/Pose6dSolver1202/workspace/姿态测量"
window = 0
sph = common.sphere(16,16,1)
plane = common.plane(12,12,1.,1.)
obj1 = common.Object3D( name="obj1", color=[1.,0.,0.])
obj1.load_from_stl(dir_root+"/models_solve/obj_1.stl")
pose = np.loadtxt(dir_root+"/results_solve/obj_1/scene_20.txt")
obj1.rmat = geo.r_to_R(pose[:3])

cube = common.Cube([0.2, 0.2, 0.2])

model = FileIO.load_model_from_stl_binary(dir_root+"/models_solve/obj_1.stl")
img = cv2.imread(dir_root+"/images_solve/cam_2/scene_20.png")
#img = cv2.resize(img, (480, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
img[:,:,3] =  img[:,:,3] // 3
imggl = Image.fromarray(img).tobytes("raw", "RGBA", 0, -1)
[height, width] = img.shape[0:2]
[near, far] = [0.001, 100]

ax = common.Axis([100, 100, 100])


from easydict import EasyDict 
cam2 = EasyDict(FileIO.load_camera_pars(dir_root+"/results_calib/cam_2/camera_pars.json"))
[fx, fy, _, _] = np.diag(cam2.intrin)
[cx, cy] = cam2.intrin[:2, 2]

Visualizer.draw_axis3d(img, cam2, rtvec=np.zeros(6), unit_length=0.1, width_line=1)
#Visualizer.draw_model3d(img, model, pose, cam2, color=(0, 255, 0))
# cv2.imshow("", img)
# cv2.waitKey(1)

persp = cv2_to_gl.intrinsic_to_perspetive(cam2.intrin, width, height, near, far)
proj = cv2_to_gl.intrinsic_to_project(cam2.intrin, width, height, near, far)
view1 = cv2_to_gl.extrinsic_to_modelview(cam2.extrin)
view2 = cv2_to_gl.rtvec_to_modelview(cam2.rvec, cam2.tvec)
isshow = 1
angle = 0
print("rvec", cam2.rvec)
print("tvec", cam2.tvec)

def InitGL():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glShadeModel(GL_SMOOTH)
    glDepthFunc(GL_ALWAYS)
    glViewport(0, 0, width, height)
    #gluPerspective(90, width/height, 0.001, 100.0)

    return

    
def DrawGLScene():

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    
    glPushMatrix()
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    set_projection_from_camera(cam2.intrin[:3, :3])
    set_modelview_from_camera(cam2.extrin[:3])
    cube.draw()
    glPopMatrix()

    
    # glPushMatrix()
    # draw_image()
    # glPopMatrix()

    glutSwapBuffers()
    time.sleep(0.1)
    #print("PROJ:\n",glGetFloatv(GL_PROJECTION_MATRIX))v
    #print("VIEW:\n",glGetFloatv(GL_MODELVIEW_MATRIX))

def draw_image():
    """使用四边形绘制背景图像"""
    
    #载入背景图像，转为OpenGL纹理
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #绑定纹理
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imggl)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    #创建四方形填充整个窗口
    
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, -1.0)
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, -1.0)
    glEnd()
    #清除纹理
    glDeleteTextures(1)
    return


def mouseButton( button, mode, x, y ):
    return

def ReSizeGLScene(Width, Height): 
    glViewport(0, 0, Width, Height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    gluPerspective(np.pi/4, Width/Height, 0.001, 1000.0)
    #glMatrixMode(GL_MODELVIEW)
    return

def cvshow():
    
    glReadBuffer(GL_FRONT)
	#从缓冲区中的读出的数据是字节数组
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.zeros((height*width*3), dtype=np.uint8)
    for i in range(0, len(data), 3):
    	#由于opencv中使用的是BGR而opengl使用的是RGB所以arr[i] = data[i+2]，而不是arr[i] = data[i]
        arr[i] = data[i+2]
        arr[i+1] = data[i+1]
        arr[i+2] = data[i]
    arr = np.reshape(arr, (height, width, 3))
    #因为opengl和OpenCV在Y轴上是颠倒的，所以要进行垂直翻转，可以查看cv2.flip函数
    cv2.namedWindow("scene", cv2.WINDOW_FREERATIO)
    cv2.flip(arr, 0, arr)
    cv2.imshow('scene', arr)
    cv2.waitKey(1000)
    isshow = 0


def set_projection_from_camera(K):
    """从照相机标定矩阵中获得视图"""
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    fovy = 2 * np.arctan(0.5 * height / fy) * 180 / np.pi
    aspect = (width * fy) / (height * fx)
    near = 0.1  #定义近的和远的裁剪平面
    far = 100.0
    gluPerspective(fovy, aspect, near, far)  #设定透视
    glViewport(0, 0, width, height)


def set_modelview_from_camera(Rt):
    """从照相机姿态中获得模拟视图矩阵"""
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  #围绕x轴将茶壶旋转90度，使z轴向上
    R = Rt[:, :3]  #获得旋转的最佳逼近
    U, S, V = np.linalg.svd(R)
    R = np.dot(U, V)
    R[0, :] = -R[0, :]  #改变x轴的符号
    t = Rt[:, 3]  #获得平移量
    M = np.eye(4) #获得4*4的模拟视图矩阵
    M[:3, :3] = np.dot(R, Rx)
    M[:3, 3] = t
    M = M.T  #转置并压平以获得列序数值
    m = M.flatten()
    glLoadMatrixf(m)  #将模拟视图矩阵替换为新的矩阵

def main():
    global window
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(500, 500) # 窗体位置
    window = glutCreateWindow("opengl")
    glutDisplayFunc(DrawGLScene)
    glutIdleFunc(DrawGLScene)
    glutMouseFunc( mouseButton )
    glutKeyboardFunc(obj1.keyboard_control)
    InitGL()
    glutMainLoop()

main()