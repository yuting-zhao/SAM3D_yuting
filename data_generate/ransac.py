import open3d as o3d
import numpy as np
import os

def fit_ground_and_colorize(input_pcd_path, output_pcd_path, distance_threshold=0.2):
    """
    拟合地面平面，地面点上色为灰色并保存新的点云文件。
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(input_pcd_path)

    # 拟合地面平面
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    a, b, c, d = plane_model
    print(f"拟合平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # 设置颜色：地面为灰色 [0.5, 0.5, 0.5]，其余保留或默认白色
    colors = np.ones((len(pcd.points), 3))  # 默认白色
    colors[:] = [1.0, 1.0, 1.0]             # 非地面为白色
    colors[inliers] = [0.5, 0.5, 0.5]       # 地面为灰色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存新的 PCD 文件
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print(f"带地面着色的点云已保存到：{output_pcd_path}")

# 示例调用
input_path = "./data_test/velodyne/000000.pcd"
output_path = "./data_test/ransac/000000.pcd"
fit_ground_and_colorize(input_path, output_path)
