

import numpy as np
import cv2

from src import Camera




def calc_depth_general_case(intrinsic1, intrinsic2, extrinsic1, extrinsic2, bbox1, bbox2, observer_idx=0):
    #print   (bbox1, bbox2   )
    fx1 = intrinsic1._fx()
    fy1 = intrinsic1._fy()
    fx2 = intrinsic2._fx()
    fy2 = intrinsic2._fy()

    cx1 = intrinsic1._cx()
    cy1 = intrinsic1._cy()
    cx2 = intrinsic2._cx()
    cy2 = intrinsic2._cy()

    c1 = extrinsic1._c()
    c2 = extrinsic2._c()

    [x1, y1, w1, h1] = bbox1
    [x1, y1] = bbox1[:2] + (bbox1[2:] / 2)
    [x2, y2, w2, h2] = bbox2
    [x2, y2] = bbox2[:2] + (bbox2[2:] / 2)

    if (w1 == 0) or (w2 == 0) or (h1 == 0) or (h2 == 0):
        return -1
    else:
        s_w = w1 / w2
        s_h = h1 / h2
        
        if observer_idx == 0:

            z1_x = fx1 * (c1[0] - c2[0]) / \
                ((x2 - cx1) * s_w - (x1 - cx1) )

            z1_y = fy1 * (c1[1] - c2[1]) / \
                ((y2 - cy1) * s_h - (y1 - cy1) )
            z1 = (z1_y + z1_x) / 2
            return [z1_x, z1_y]

        elif observer_idx == 1:
            z2_x = fx2 * (c2[0] - c1[0]) / \
                ((x1 - cx2) / s_w - (x2 - cx2) )

            z2_x = fy2 * (c2[1] - c1[1]) / \
                ((y1 - cy2) / s_h - (y2 - cy2) )
            z2 = (z2_x + z2_x) / 2
            return [z2_x, z2_x]

def calc_depth_dz_equals_zero(intrinsic1, intrinsic2, extrinsic1, extrinsic2, bbox1, bbox2, observer_idx=0):
    fx = intrinsic1._fx()
    fy = intrinsic1._fy()

    cx = intrinsic1._cx()
    cy = intrinsic1._cy()

    c1 = extrinsic1._c()
    c2 = extrinsic2._c()

    [x1, y1, w1, h1] = bbox1
    [x2, y2, w2, h2] = bbox2

    if (w1 == 0) or (w2 == 0) or (h1 == 0) or (h2 == 0):
        return -1
    else:
        s_w = w1 / w2
        s_h = h1 / h2
        
        if observer_idx == 0:
            s = (s_w + s_h) / 2

            z1_x = fx * (c1[0] - c2[0]) / (x2 - x1)
                
            z1_y = fy * (c1[1] - c2[1]) / (y2 - y1)
                

            z1 = (z1_y + z1_x) / 2
            return [z1_x, z1_y]

        elif observer_idx == 1:
            s = (1/s_w + 1/s_h) / 2

            z2_x = fx * (c2[0] - c1[0]) / (x1 - x2)

            z2_y = fy * (c2[1] - c1[1]) / (y1 - y2)

            z2 = (z2_x + z2_y) / 2
            return [z2_x, z2_y]