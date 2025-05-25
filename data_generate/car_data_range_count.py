import matplotlib
matplotlib.use('Agg')

import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置你的标注文件夹路径
label_folder = "/home/zhaoyuting/DAIR/single-infrastructure-side/label/virtuallidar"

xyz_list = []
lwh_list = []

all_files = [f for f in os.listdir(label_folder) if f.endswith(".json")]

for filename in tqdm(all_files, desc="Processing label files"):
    file_path = os.path.join(label_folder, filename)
    with open(file_path, "r") as f:
        try:
            annotations = json.load(f)
        except:
            continue

        for obj in annotations:
            if obj.get("type") == "Car":
                loc = obj.get("3d_location", {})
                dim = obj.get("3d_dimensions", {})

                try:
                    x = float(loc["x"])
                    y = float(loc["y"])
                    z = float(loc["z"])

                    h = float(dim["h"])
                    w = float(dim["w"])
                    l = float(dim["l"])

                    xyz_list.append([x, y, z])
                    lwh_list.append([l, w, h])
                except:
                    continue

xyz_array = np.array(xyz_list)
lwh_array = np.array(lwh_list)

def print_range(name, array):
    min_vals = array.min(axis=0)
    max_vals = array.max(axis=0)
    print(f"{name} 范围：")
    print(f"  最小值：x={min_vals[0]:.3f}, y={min_vals[1]:.3f}, z={min_vals[2]:.3f}")
    print(f"  最大值：x={max_vals[0]:.3f}, y={max_vals[1]:.3f}, z={max_vals[2]:.3f}")
    print()

if xyz_array.shape[0] > 0:
    print_range("3D位置 (xyz)", xyz_array)

if lwh_array.shape[0] > 0:
    print_range("3D尺寸 (lwh)", lwh_array)

# ===================
# 保存直方图到文件
# ===================
def save_histograms(data, labels, title_prefix, filename, bins=30):
    plt.figure(figsize=(15, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.hist(data[:, i], bins=bins, color='skyblue', edgecolor='black')
        plt.xlabel(labels[i])
        plt.ylabel("频数")
        plt.title(f"{title_prefix} - {labels[i]}")
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"已保存: {save_path}")

if xyz_array.shape[0] > 0:
    save_histograms(xyz_array, ["x", "y", "z"], "3D位置分布", "car_xyz_hist.png")

if lwh_array.shape[0] > 0:
    save_histograms(lwh_array, ["length", "width", "height"], "3D尺寸分布", "car_lwh_hist.png")