
from easydict import EasyDict
import numpy as np
import cv2
import json
import os

from src import geometries
from src import visualizer


class CalibratorByDLT(object):
    def __init__(self, unit_length_in_meter=1) -> None:
        self.unit_length_in_meter = unit_length_in_meter
        self.solve_perspective_matrix_3d_to_2d = geometries.solve_projection_matrix_3d_to_2d
        # RQ decompose or normal
        self.decompose_intrin_extrin_from_projection_mat = geometries.decompose_projection_mat
        self.R2r = geometries.R_to_r
        self.T2t = geometries.T_to_t
        return

    def set_points3d(self, points3d: np.ndarray):
        self.points3d_real = points3d * self.unit_length_in_meter
        return

    def set_default_points3d(self, choosen_indexes_z0=range(4), choosen_indexes_z1=range(4)):
        grid = np.mgrid[0:2, 0:2, 0:2]
        grid = grid.reshape((3, 4, 2)).T
        grid = grid[:, [0, 2, 1, 3]]
        a = grid[0, choosen_indexes_z0, :]
        b = grid[1, choosen_indexes_z1, :]
        self.set_points3d(np.vstack([a, b]))
        return

    def set_points2d(self, points2d: np.ndarray):
        self.points2d_obj = points2d
        return
    
    def solve(self):
        M = self.solve_perspective_matrix_3d_to_2d(self.points3d_real, self.points2d_obj, method="ols")
        [mat_intrin, mat_extrin] = self.decompose_intrin_extrin_from_projection_mat(M)
 
        rvec = self.R2r(mat_extrin)
        tvec = self.T2t(mat_extrin)
        self.camera_pars = {}
        self.camera_pars["intrin"] = mat_intrin
        self.camera_pars["extrin"] = mat_extrin
        self.camera_pars["rvec"] = rvec
        self.camera_pars["tvec"] = tvec
        return

    def outprint(self):
        print()
        print("intrin:\n", self.camera_pars["intrin"])
        print("extrin:\n", self.camera_pars["extrin"])
        print("rvec:\n", self.camera_pars["rvec"])
        print("tvec:\n", self.camera_pars["tvec"])
        print()
        return
    
    def run(self):
        print("\n开始标定...")
        self.solve()
        self.outprint()
        return


class CalibratorArucoByDLT(CalibratorByDLT):
    def __init__(self, unit_length_in_meter=1, offset_length_in_meter=0):
        super().__init__(unit_length_in_meter)
        self.offset_length_in_meter = offset_length_in_meter
        self.__set_points3d()
        return

    def __set_points3d(self):
        grid = np.mgrid[0:2, 0:2, 0:2]
        grid = grid.reshape((3, 4, 2)).T
        offset_scale = np.array([
            [[ 3,  1,  0],
             [-1,  3,  0],
             [ 1, -3,  0],
             [-3, -1,  0]],
            [[ 1,  3,  0],
             [-3,  1,  0],
             [ 3, -1,  0],
             [-1, -3,  0]]
        ])
        points3d = np.zeros((6, 2, 4, 3))
        points3d[0] = grid.copy() * self.unit_length_in_meter + self.offset_length_in_meter / 2 * offset_scale
        points3d[1] = grid.copy() * self.unit_length_in_meter + self.offset_length_in_meter / 2 * np.flip(offset_scale, axis=0)
        points3d[2] = grid.copy() * self.unit_length_in_meter + self.offset_length_in_meter / 2 * np.flip(offset_scale, axis=0)
        self.set_points3d()
        return

    


def mouse_event(event, x, y, flags, param):
    # print('[{},{}]'.format(x, y))  # 坐标，原点在左上角

    '''
    if flags == cv2.EVENT_FLAG_ALTKEY:
        print('摁住Alt')
    if flags == cv2.EVENT_FLAG_CTRLKEY:
        print('摁住Ctrl')
    if flags == cv2.EVENT_FLAG_SHIFTKEY:
        print('摁住Shift')

    if flags == cv2.EVENT_FLAG_LBUTTON:
        print('摁住左键')
    if flags == cv2.EVENT_FLAG_MBUTTON:
        print('摁住中键')
    if flags == cv2.EVENT_FLAG_RBUTTON:
        print('摁住右键')
    '''

    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('左键双击', "x: {}, y: {}".format(x, y))
        points2d.append([x, y])
    if event == cv2.EVENT_MBUTTONDBLCLK:
        print('中键双击')
    if event == cv2.EVENT_RBUTTONDBLCLK:
        print('右键双击')

    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        print('左键击下')
    if event == cv2.EVENT_LBUTTONUP:
        print('左键弹起')
    if event == cv2.EVENT_MBUTTONDOWN:
        print('中键击下')
    if event == cv2.EVENT_MBUTTONUP:
        print('中键弹起')
    if event == cv2.EVENT_RBUTTONDOWN:
        print('右键击下')
    if event == cv2.EVENT_RBUTTONUP:
        print('右键弹起')
    '''

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            print('向前滚动')
        else:
            print('向后滚动')
    if event == cv2.EVENT_MOUSEHWHEEL:
        if flags > 0:
            print('向左滚动')  # 按住Alt
        else:
            print('向右滚动')



if __name__ == '__main__':
    # input args:
    idx_cam  = 0
    dir_root         = "test_405"
    dir_folder       = "test_405/marker/" + str(idx_cam)
    dir_outfolder    = "test_405/calib/" + str(idx_cam)
    name_input_img   = "0.png"
    unit_length = 0.06 

    name_win = "Calibrator by DLT"
    cv2.namedWindow(name_win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(name_win, mouse_event)  # 窗口与回调函数绑定
    
    [_, name_subdir] = os.path.split(dir_folder)
    [prefix_img, _]  = os.path.splitext(name_input_img)
    name_output_img  = "{}_calib.jpg".format(prefix_img)
    name_output_json = "{}_campars.json".format(prefix_img)
    name_pts_json = "{}_points.json".format(prefix_img)

    pth_input_img   = os.path.join(dir_folder, name_input_img)
    pth_output_img  = os.path.join(dir_outfolder, name_output_img)
    pth_output_json = os.path.join(dir_root, "calib", name_subdir, name_output_json)
    pth_pts_json = os.path.join(dir_outfolder, "calib", name_subdir, name_pts_json)

    img = cv2.imread(pth_input_img)
    points2d = []
    while True:
        cv2.imshow(name_win, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 摁下q退出
            break
    points2d = np.array(points2d, dtype=float)

    calibrator = CalibratorByDLT(unit_length)
    calibrator.set_points2d(points2d)
    calibrator.set_default_points3d([0, 1, 2], [0, 1, 2])
    calibrator.run()

    visualizer.draw_points3d_with_texts(img, calibrator.points3d_real, np.zeros(6), calibrator.camera_pars)
    visualizer.draw_points2d(img, points2d)
    visualizer.draw_axes3d(img, calibrator.camera_pars, unit_length=unit_length)
    cv2.imwrite(pth_output_img, img)
    cv2.imshow(name_win, img)
    cv2.waitKey()

    os.makedirs(os.path.join(dir_root, "calib", name_subdir)) if (not os.path.exists(os.path.join(dir_root, "calib", name_subdir))) else 1
    with open(pth_output_json, "w") as f:
        for key in calibrator.camera_pars.keys():
            data = calibrator.camera_pars[key]
            if type(data) is np.ndarray:
                calibrator.camera_pars[key] = data.tolist()
        json.dump(calibrator.camera_pars, f, indent=4)
        print(pth_output_json)
    cv2.destroyAllWindows()
    print()
    