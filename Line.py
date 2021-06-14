
from typing import *
import numpy as np
import cv2

from Point import Point

class Line():
    def __init__(self):
        super().__init__()
        self.mat = np.zeros((3,))
        return

    def set_by_2pts(self, points: List[Point]):
        pt_s = points[0]
        pt_t = points[1]
        Xl = 0

