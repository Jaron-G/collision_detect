# Point Cloud Collision Detection Using Open3D

intersection_method文件中函数contact_detect实现通过场景点云与夹爪点云取交集来检测夹爪与场景是否发生碰撞

* param matched_matrix: 相机坐标系下配准后的变换矩阵
* param coincide_num_points: 重合的点数阈值
* param dist_max: 判定重合点之间距离的阈值，即相邻点距离多近会判断为重合点
* return: 是否发生碰撞

cropping_method文件中函数contact_detect实现将场景点云变换到夹爪坐标系下，通过夹爪finger大小的包围盒进行点云裁减点获取碰撞点数来检测夹爪与场景是否发生碰撞

* param matched_matrix: 相机坐标系下配准后的变换矩阵
* param coincide_num_points: 重合的点数阈值
* return: 是否发生碰撞

包内文件

* matrix.txt为手眼变换矩阵
* scene_gazebo.ply为场景点云
* main.py为测试程序
* gripper_2F85.STL为夹爪模型
