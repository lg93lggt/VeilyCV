
import numpy as np
import cv2
import open3d as o3d
import sklearn
import scipy



# 功能：iss方法获取特征点
# 输入：
#     data: 输入数据，np.array数组
#     gamma21: 判断特征点的参数
#     gamma32: 判断特征点的参数
#     radius: KDTree的邻近搜索半径
#     NMS_radius: 非极大抑制(NMS)中计算加权协方差矩阵的邻域半径
#     max_num: 最大迭代次数，同时自然也是提取出的最多的特征点数
# 输出：
#     keypoints_after_NMS: 非极大抑制处理后的关键点
def ISS(vertexes,gamma10,  gamma21, radius, nms=True, radius_nms=2, max_num=100):

    """
    1. Compute a weighted array w_vec for each point inversely related to the number of points in its spherical neighborhood of radius.
    2. Compute a weighted scatter matrix cov_mat.
    """
    eps = 1e-8

    n_points = vertexes.shape[0]

    # compute r indensity
    leaf_size = 32

    kdtree = sklearn.neighbors.KDTree(vertexes, metric="euclidean")
    row_idxes_in_radius = kdtree.query_radius(vertexes, radius, return_distance=False)
    counts_in_radius = kdtree.query_radius(vertexes, radius, return_distance=False, count_only=True)
    w_vec = 1 / (counts_in_radius + eps)

    idxes_kpts = []  # 首先定义keypoints集合
    e_min_kpts = []  # 定义最小特征值集合 TODO 何用
    for [idx_i, n] in enumerate(counts_in_radius):
        if n == 1:
            continue
        else:
            cov = np.zeros((3, 3))
            idxes_j = row_idxes_in_radius[idx_i]
            for idx_j in idxes_j:
                [v_i, v_j] = vertexes[[idx_i, idx_j], :]
                delta = (v_j - v_i).reshape((-1, 1))
                cov += delta @ delta.T
            cov = cov / np.sum(w_vec)
            [_, eig_value, _] = np.linalg.svd(cov)
            if (eig_value[1] / (eig_value[0] + eps) < gamma10) and (eig_value[2] / (eig_value[1] + eps) < gamma21):
                idxes_kpts.append(idx_i)
                e_min_kpts.append(eig_value[2])

    kpts_iss = vertexes[idxes_kpts]
    tree = scipy.spatial.KDTree(kpts_iss, leaf_size)
    radius_neighbor = tree.query_ball_point(kpts_iss, radius_nms)
    return  kpts_iss


def ISS2(data, gamma21, gamma32, radius, NMS_radius, max_num=100):
    # iss: Intrinsic Shape Signatures,
    # main idea: Keypoints are those have large 3D point variations in their neighborhood
    print("-"*10, "start to do iss", data.shape[0], "points", "-"*10)
    # 利用scipy中的KDTree来找邻域，TODO 尝试一下自己写这几个查找树
    leaf_size = 32
    # leaf_size这个参数是算法切换到暴力破解的点数
    # The number of points at which the algorithm switches over to brute-force.
    tree = scipy.spatial.KDTree(data, leaf_size)
    # tree.query_ball_point(): Find all points within distance r of point(s) x.
    # If x is a single point, returns a list of the indices of the neighbors of x.
    # If x is an array of points, returns an object array of shape tuple
    # containing lists of neighbors.
    # 该函数返回的是索引
    radius_neighbor = tree.query_ball_point(data, radius)

    print("-" * 10, "start to search keypoints", '-' * 10)
    keypoints = []  # 首先定义keypoints集合
    min_feature_value = []  # 定义最小特征值集合 TODO 何用
    for index in range(len(radius_neighbor)):
        neighbor_idx = radius_neighbor[index]   # 得到第一个点的邻近点索引
        neighbor_idx.remove(index)  # 把自己去掉
        # 如果一个点没有邻近点，直接跳过 TODO 邻近点少的是否也可以直接跳过不要了
        if len(neighbor_idx)==0:
            continue

        # 计算权重矩阵
        weight = np.linalg.norm(data[neighbor_idx] - data[index], axis=1)
        weight[weight == 0] = 0.001  # 避免除0的情况出现
        weight = 1 / weight

        # 直接循环求解，计算加权协方差矩阵
        cov = np.zeros((3, 3))
        # 这里需要加个np.newaxis，变成N*3*1，这样tmp[i]才能是3*1的矩阵
        tmp = (data[neighbor_idx] - data[index])[:, :, np.newaxis]
        for i in range(len(neighbor_idx)):
            cov += weight[i]*tmp[i].dot(tmp[i].transpose())
        cov /= np.sum(weight)

        # 或者用下边的方法计算加权协方差矩阵
        '''
        tmp = (data[neighbor_idx] - data[index])[:, :, np.newaxis]  # N,3,1
        cov = np.sum(weight[:, np.newaxis, np.newaxis] *
                     (tmp @ tmp.transpose(0, 2, 1)), axis=0) / np.sum(weight)
        '''

        # 做特征值分解，SVD
        s = np.linalg.svd(cov, compute_uv=False)    # 不必计算u和vh，只计算特征值即可，默认为True
        # 根据特征值判断是否为特征点
        if (s[1]/s[0] < gamma21) and (s[2]/s[1] < gamma32):
            keypoints.append(data[index])
            min_feature_value.append(s[2])
    print("search keypoints finished", keypoints.__len__(), "points")   # print可以这样使用

    # NMS step
    # 非极大值抑制（Non-Maximum Suppression，NMS），顾名思义就是抑制不是极大值的元素，可以理解为局部最大搜索。
    # 也就是找到局部中最像特征值的那个点
    print("-"*10, "NMS to filter keypoints", "-"*10)
    keypoints_after_NMS = []
    leaf_size = 10  # 又来了一个leaf_size
    nms_tree = scipy.spatial.KDTree(keypoints, leaf_size)
    index_all = [i for i in range(len(keypoints))]
    for iter in range(max_num):
        # 找到s2特征值集合中最大点的索引
        max_index = min_feature_value.index(max(min_feature_value))
        # max_index = np.argmax(min_feature_value)
        tmp_point = keypoints[max_index]
        # 找到s2特征值集合中最大点的邻近
        del_indexs = nms_tree.query_ball_point(tmp_point, NMS_radius)
        # 删去找到的邻近，之后要只保留最大的s2特征值点对应的keypoint
        for del_index in del_indexs:
            if del_index in index_all:
                del min_feature_value[index_all.index(del_index)]   # 删去对应的特征值
                del keypoints[index_all.index(del_index)]           # 删去对应的关键点
                del index_all[index_all.index(del_index)]           # 删去对应的索引
        # NSM:保留最大的s2特征值点对应的keypoints
        keypoints_after_NMS.append(tmp_point)
        # 如果此时keypoints已经为0了，那就可以break了
        if len(keypoints) == 0:
            break
    print("NMS finished,find ", len(keypoints_after_NMS), " points")

    return np.array(keypoints_after_NMS)
    
def test(pth_input_mesh):
    mesh = o3d.io.read_triangle_mesh(pth_input_mesh)
    pts = np.array(mesh.vertices)
    #keypoint = ISS(pts, gamma10=0.6, gamma21=0.6, radius=0.15, radius_nms=0.3, max_num=5000)
    keypoint = ISS2(pts, 0.6, 0.6, 0.15, 0.3, 5000)

    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
    key_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(keypoint))

    key_view.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([key_view, pc_view, mesh])

if __name__ == "__main__":
    pth_input_mesh = "/media/veily3/Data/LiGan/Pose6dSolver-pyqt/姿态测量/models_solve/obj_1.stl"
    test(pth_input_mesh)