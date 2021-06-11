
from  icecream import ic
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import copy
from spatialmath import *
from spatialmath.base import rotz

pth_mesh_src = Path("/media/veily3/data_ligan/匹克脚部数据集/gt/ligan/right.stl")
mesh = o3d.io.read_triangle_mesh(str(pth_mesh_src))
mesh.rotate(rotz(90, 'deg'))
t = np.zeros(3)
t[1] = -mesh.get_center()[1]
mesh.translate(t)
o3d.io.write_triangle_mesh(str(pth_mesh_src.parent / (pth_mesh_src.stem + ".ply")), mesh)

