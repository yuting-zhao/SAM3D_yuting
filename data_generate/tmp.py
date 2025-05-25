# import open3d as o3d
# import numpy as np

# pcd = o3d.io.read_point_cloud("/home/zhaoyuting/SAM3D_2/SAM3D_yuting/data_generate/data_test/velodyne/000000.pcd")
# points = np.asarray(pcd.points)

# x_min, y_min, z_min = points.min(axis=0)
# x_max, y_max, z_max = points.max(axis=0)

# print(f"X range: {x_min:.2f} ~ {x_max:.2f}")
# print(f"Y range: {y_min:.2f} ~ {y_max:.2f}")
# print(f"Z range: {z_min:.2f} ~ {z_max:.2f}")

from pypcd import pypcd
import numpy as np

def get_xyz_intensity_range_pypcd(pcd_file):
    # 读取完整 pcd 数据
    pc = pypcd.PointCloud.from_path(pcd_file)

    # 检查字段是否存在
    fields = pc.fields
    print("PCD fields:", fields)

    # 获取 xyz 范围
    x, y, z = pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z']
    print("X range: [{:.3f}, {:.3f}]".format(np.min(x), np.max(x)))
    print("Y range: [{:.3f}, {:.3f}]".format(np.min(y), np.max(y)))
    print("Z range: [{:.3f}, {:.3f}]".format(np.min(z), np.max(z)))

    # 获取 intensity 范围（如果有）
    if 'intensity' in fields:
        i = pc.pc_data['intensity']
        print("Intensity range: [{:.3f}, {:.3f}]".format(np.min(i), np.max(i)))
    else:
        print("No intensity field found in PCD.")

# 示例调用
get_xyz_intensity_range_pypcd("/home/zhaoyuting/SAM3D_2/SAM3D_yuting/data_generate/data_test/velodyne/000000.pcd")

