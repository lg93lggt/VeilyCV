
import numpy as np
import cv2
import json
from  easydict import  EasyDict
from time import time

t0 = time()

# 数据初始化及预处理
eps = 1E-10
xlim, ylim ,zlim = 0.4, 0.4, 0.4
dx, dy, dz = 0.002, 0.002, 0.002
gird = np.mgrid[0:xlim:dx, 0:ylim:dy, 0:zlim:dz]
result = np.zeros(gird.shape[1:])

# 加载内参
intrinsic = np.loadtxt('./data/intrinsic.txt',dtype=np.float32)

for x in range(1,15):   # 14个外参  14个mask
    extrinsic = np.loadtxt('./data/extrinsic'+str(x)+'.txt',dtype=np.float32)
    mask = cv2.imread("./masks/"+str(x)+".jpg", cv2.IMREAD_GRAYSCALE)
    voxels = np.ones(gird.shape[1:])
    # 三个for循环对应着200x200x200
    for [i, _] in enumerate(np.arange(0, xlim, dx)):
        for [j, _] in enumerate(np.arange(0, ylim, dy)):
            for [k, _] in enumerate(np.arange(0, zlim, dz)):
                pt = np.ones(4)
                pt[:3] = gird[:, i, j, k]
                
                # 投影
                _p = intrinsic @ extrinsic @ pt.reshape((4, 1))
                _p = _p / (_p[-1] + eps)
                # 四舍五入
                [col, row, _] = np.round(_p).astype(np.int).reshape(3)

                # 因为有越界情况，所以需要做越界判断
                if row<1440 and col <1080:
                    if mask[row, col] == 0:
                        voxels[i, j, k] = 0
                else:
                    voxels[i, j, k] = 0 
    # 14个voxels累加           
    result += voxels

# 阈值(12)判断,>12置1，<12置0
for i in range(200):
    for j in range(200):
        for k in range(200):
            if result[i,j,k] > 12:
                result[i,j,k] = 1
            else:
                result[i,j,k] = 0
# 保存最后的结果
np.save("result002.npy",result)

t1 = time()
# 打印耗时
print("200x200x200耗时：%fs"%(t1-t0))
