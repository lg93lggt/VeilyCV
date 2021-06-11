

import numpy as np
import cv2
import open3d as o3d
import time
import matplotlib.pyplot as plt
import argparse
from easydict import EasyDict


"""
Hyper params
"""
FRONT = np.array([0.4, 0.4, 0.8])
UP    = np.array([0, 1., 0])
AT    = np.array([0.5, 0.5, 0])
W     = 640
H     = 480
ZOOM  = 1.5


def to_homo(P, axis=0):
    """
    @ axis: 
        if axis==0, row main vector
        if axis==1, col main vector
    """
    shape = P.shape
    if   axis==0:
        addon_col = np.ones(shape[0])
        return np.vstack((P.T, addon_col)).T
    elif axis==1:
        addon_row = np.ones(shape[1])
        return np.vstack((P, addon_row))


def params_o3d_to_intrinsic_cv(fov_rad, height, width):
    fx = fy = (height / 2) / np.tan(fov_rad / 2)
    cx = width  / 2 - 0.5
    cy = height / 2 - 0.5
    K = np.diag([fx, fy, 1])
    K[0, 2] = cx
    K[1, 2] = cy
    return K

def intrinsic_cv_to_params_o3d(K: np.ndarray):
    pass
    return K

def params_o3d_to_extrinsic_cv(fov_rad, front, up, at, zoom):
    [at, eye, up] = params_o3d_to_params_opengl(fov_rad, front, up, at, zoom)
    front = front / np.linalg.norm(front)
    right    = np.cross(up, front)
    right    = right / np.linalg.norm(right)
    R = np.array([right, -up, -front])
    t = -R @ eye.reshape((1, -1)).T

    M = np.eye(4)
    M[:3,  :3] = R
    M[:3, 3:4] = t
    return M


def params_opengl_to_extrinsic_cv(fov_rad, at, eye, up, zoom):
    extent   = np.ones(3)
    ratio    = extent * zoom
    distance = ratio / np.tan(fov_rad/2)
    front = (eye - at) / distance
    right    = np.cross(up, front)
    right    = right / np.linalg.norm(right)
    R = np.array([right, -up, -front])
    t = -R @ eye.reshape((1, -1)).T

    M = np.eye(4)
    M[:3,  :3] = R
    M[:3, 3:4] = t
    return M


def params_o3d_to_params_opengl(fov_rad, front, up, at, zoom):
    front = front / np.linalg.norm(front)
    up    = up    / np.linalg.norm(up)

    extent   = np.ones(3)
    ratio    = extent * zoom
    distance = ratio / np.tan(fov_rad/2)
    right    = np.cross(up, front)
    right    = right / np.linalg.norm(right)
    up       = np.cross(front, right) # ??? TODO make R belongs to SO3, det(R) ~= 1
    eye      = at + front * distance
    print(at, eye, up)
    return [at, eye, up]


"""
Define a cube:
@ verts:
    array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]], dtype=np.float)
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
verts = np.mgrid[0:2, 0:2, 0:2].reshape((3, -1)).T

faces = np.array(
    [[0, 2, 1],
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
     [3, 2, 6]], 
    dtype=int
)

edges = np.array(
    [[0, 1],
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
     [3, 7]],
    dtype=int
)
colors_edge = np.ones((12, 3))
colors_edge[[0, 1, 8]] = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

cube = EasyDict({})
cube.lineset            = o3d.geometry.LineSet()
cube.lineset.points     = o3d.utility.Vector3dVector((verts.copy() - 0.5)*0.1)
cube.lineset.lines      = o3d.utility.Vector2iVector(edges)
cube.lineset.colors     = o3d.utility.Vector3dVector(colors_edge)

cube.mesh               = o3d.geometry.TriangleMesh()
cube.mesh.vertices      = o3d.utility.Vector3dVector((verts.copy() - 0.5)*0.1)
cube.mesh.triangles     = o3d.utility.Vector3iVector(faces)
cube.mesh.vertex_colors = o3d.utility.Vector3dVector(np.flip(verts.copy(), axis=0))


if __name__ == "__main__":

    """
    Hyper params
    FRONT = np.array([0.4, 0.4, 0.8])
    UP    = np.array([0, 1., 0])
    AT    = np.array([0.5, 0.5, 0])
    W     = 640
    H     = 480
    ZOOM  = 1.5
    """

    parser = argparse.ArgumentParser(
        description="Render a 1x1x1  cube by open3d.")
    parser.add_argument(
        "-W", 
        type=int, 
        default=640,
        help="image width, int"
    )
    parser.add_argument(
        "-UP", 
        type=float, 
        nargs="+",
        default=[0., 1., 0.],
        help="up array, float"
    )
    parser.add_argument(
        "-FRONT", 
        type=float, 
        nargs="+",
        default=[0.4, 0.4, 0.8],
        help="front array, float"
    )
    parser.add_argument(
        "-AT", 
        type=float, 
        nargs="+", 
        default=[0.5, 0.5, 0.],
        help="at array, float"
    )
    parser.add_argument(
        "-ZOOM", 
        type=float, 
        default=1.5,
        help="zoom ratio, float"
    )
    args = parser.parse_args()

    ratio_wh = 640 / 480
    FRONT = np.array(args.FRONT)
    UP    = np.array(args.UP)
    AT    = np.array(args.AT)
    ZOOM  = args.ZOOM
    W     = args.W
    H = int(W / ratio_wh)

    """
    Open3d Renderer
    """
    #vis = o3d.visualization.Visualizer()
    vis = o3d.visualization.Visualizer()
    vis.create_window("o3d", width=W, height=H)
    time.sleep(1)
    vis.add_geometry(cube)


    ctl = vis.get_view_control()
    ctl.set_front(FRONT)
    ctl.set_up(UP)
    ctl.set_lookat(AT)
    ctl.set_zoom(ZOOM)

    opt = vis.get_render_option()
    opt.background_color = np.zeros(3)
    vis.poll_events()
    #vis.run()

    """
    Plugin to OpenCV
    """

    img_rgb   = vis.capture_screen_float_buffer()
    img_depth = vis.capture_depth_float_buffer()
    img_rgb   = (np.asarray(img_rgb  ) *  255).astype(np.uint8)
    img_depth = (np.asarray(img_depth) * 1000).astype(np.uint16)
    img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    intrin = params_o3d_to_intrinsic_cv(fov_rad=np.pi*ctl.get_field_of_view()/180, height=H, width=W)
    extrin = params_o3d_to_extrinsic_cv(fov_rad=np.pi*ctl.get_field_of_view()/180, up=UP, front=FRONT, at=AT, zoom=ZOOM)

    [R, T] = [np.eye(4), np.eye(4)]
    R[:3, :3] = extrin[:3, :3]
    T[:3,  3] = extrin[:3,  3]
    # to_homo(verts, axis=0)
    P3d = extrin @ to_homo(verts.T, axis=1)
    P3d = P3d / P3d[-1]

    P2d = intrin @ P3d[:3]
    P2d = P2d / P2d[-1]
    P2d_plot = np.round(P2d[:2]).astype(int).T
    img_pro = img_bgr.copy()
    delta = 20
    for [idx, p] in enumerate(P2d_plot):  
        try:
            img_pro = cv2.circle(img_pro, tuple(p), 3, (178, 110, 242))
            pth = "./error_p{:d}_{:d}x{:d}.png".format(idx + 1, H, W)
            cv2.imwrite(pth, img_pro[p[1]-delta:p[1]+delta, p[0]-delta:p[0]+delta, :])
            print("imwrite: " + pth)
        except:
            continue

    cv2.imwrite("./raw_{:d}x{:d}.png".format(H, W), img_bgr)
    cv2.imwrite("./project_{:d}x{:d}.png".format(H, W), img_pro)
    cv2.imwrite("./depth_{:d}x{:d}.png".format(H, W), img_depth)
    print("imwrite: ./project_{:d}x{:d}.png".format(H, W))
    vis.clear_geometries()
    vis.destroy_window()
    time.sleep(5)

