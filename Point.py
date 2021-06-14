
from typing import *
import numpy as np
import cv2

class Points():
    SUPPORT_TYPES = {
        "np.ndarray": init_by_array, 
        "dict":       init_by_dict,
        "list":       init_by_list
    }

    def __init__(self, input_points):

        print(type(input_points))
            
        self.pts = input_points
        self.len = self.pts.size // 3
        super().__init__()
        return

    def __len__(self):
        return  self.len

    def to_homo(self):
        if self.pts.shape == (2, ):
            pts_tmp = np.ones((3, ))
            pts_tmp[:2] = self.pts[:]
            self.pts = pts_tmp
        return self.pts

# class Point(Points):
#     def __init__(self, input_point):
#         super().__init__(input_point)
#         return

#     def 

if __name__ == "__main__":
    pts = np.array([[200, 300],
                    [100, 500]])
    p = Points(pts)
    p.to_homo()
    print()

