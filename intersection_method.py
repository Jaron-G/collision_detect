import open3d as o3d
import numpy as np
import transforms3d as tfs
import time


def r_t_to_homogeneous_matrix(R, T):
    R1 = np.vstack([R, np.array([0, 0, 0])])
    T1 = np.vstack([T, np.array([1])])
    HomoMtr = np.hstack([R1, T1])
    return HomoMtr


def contact_detect(matched_matrix, coincide_num_points: int, dist_max=5) -> bool:
    """
    :param matched_matrix: 相机坐标系下配准后的变换矩阵
    :param coincide_num_points: 重合的点数阈值
    :param dist_max: 判定重合点之间距离的阈值，即相邻点距离多近会判断为重合点
    :return: 是否发生碰撞
    """
    start = time.time()
    hand_eye_matrix = np.loadtxt('matrix.txt')
    scene_point_cloud = o3d.io.read_point_cloud("scene_gazebo.ply", remove_nan_points=True,
                                                remove_infinite_points=True)  # 原始点云
    gripper_mesh = o3d.io.read_triangle_mesh("gripper_2F85.STL")
    gripper_mesh.compute_vertex_normals()
    gripper_point_cloud = gripper_mesh.sample_points_poisson_disk(number_of_points=4096)

    # 创建坐标系，用于可视化
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0, 0, 0])

    # 调整夹爪竖直向下，相机坐标系
    rotate_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    gripper_point_cloud.rotate(rotate_matrix)

    # 将夹爪变换到相机坐标系下模型抓取原点
    transformation_matrix = matched_matrix
    gripper_point_cloud.transform(transformation_matrix)

    # 将场景和夹爪转换为机器人基坐标系下
    gripper_point_cloud.transform(hand_eye_matrix)
    scene_point_cloud.transform(hand_eye_matrix)

    # 由于ply格式的夹爪模型原点不位于夹爪中心，需要在这里进行调整
    translation_vector = np.array([10, -20, 0])
    gripper_point_cloud.translate(translation_vector)

    # 移动夹爪到配置点
    r_vec = np.array([[0], [0], [0]])
    pR_matrix = tfs.euler.euler2mat(r_vec[2], r_vec[1], r_vec[0], 'szyx')

    t_vec = np.array([[14], [20], [65]])  # 重合
    # t_vec = np.array([[14], [20], [100]])# 未重合

    grasp_config_in_base = r_t_to_homogeneous_matrix(pR_matrix, t_vec)
    gripper_point_cloud.transform(grasp_config_in_base)

    # o3d.visualization.draw_geometries([scene_point_cloud, gripper_point_cloud, axes], mesh_show_back_face=False)

    # 点云的相交、求异
    # 建立原始点云数据的kd-tree
    kd_tree = o3d.geometry.KDTreeFlann(scene_point_cloud)
    # pts_idx两片点云中相同部分的索引
    pts_idx = []
    # k:k-nn的搜索参数，搜索另外一片点云中距离它的最近点
    k = 1
    # 得到的点个数
    points = np.array(gripper_point_cloud.points)
    pointNum = points.shape[0]

    # 遍历点云
    for i in range(0, pointNum):
        # k:返回点个数
        # idx:返回点索引
        # dist:返回点距离
        [k, idx, dist] = kd_tree.search_knn_vector_3d(knn=k, query=gripper_point_cloud.points[i])
        if dist[0] < dist_max:  # 距离小于阈值，则认为是相同部分的点云
            pts_idx.append(i)

    same_part = gripper_point_cloud.select_by_index(pts_idx)
    diff_part = gripper_point_cloud.select_by_index(pts_idx, invert=True)

    same_part.paint_uniform_color([1, 0, 0])  # 重合点云为红色
    diff_part.paint_uniform_color([0, 0, 1])  # 未重合点云为蓝色

    print("重合的点云数量：", len(pts_idx))

    end = time.time()
    running_time = end - start
    print('time cost : %.5f sec' % running_time)

    o3d.visualization.draw_geometries([same_part, diff_part], mesh_show_back_face=False)
    print(coincide_num_points)
    if len(pts_idx) > coincide_num_points:
        return True
    else:
        return False


if __name__ == '__main__':
    matched_matrix = np.array([[7.85797448e-01, -6.06924608e-01, -1.19016350e-01, -2.34288674e+02],
                               [6.14022561e-01, 7.88620385e-01, 3.24681746e-02, 2.01817835e+01],
                               [7.41529856e-02, -9.85921327e-02, 9.92361288e-01, 9.78890064e+02],
                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_max = 10
    coincide_num_points = 80
    is_collided = contact_detect(matched_matrix, coincide_num_points, dist_max)

    print(is_collided)
    if is_collided:
        print("点云重合，发生碰撞")
    else:
        print("正常执行抓取")
