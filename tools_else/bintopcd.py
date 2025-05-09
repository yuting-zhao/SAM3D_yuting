import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

def bin_to_pcd(bin_file, pcd_file):
    try:
        raw = np.fromfile(bin_file, dtype=np.float32)
        if raw.size % 4 == 0:
            points = raw.reshape(-1, 4)
            coords = points[:, :3]
        elif raw.size % 3 == 0:
            points = raw.reshape(-1, 3)
            coords = points
        else:
            raise ValueError(f"Invalid point count: {raw.size}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        o3d.io.write_point_cloud(pcd_file, pcd, write_ascii=True)

    except Exception as e:
        print(f"[Skipped] {bin_file}: {e}")

def batch_convert(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bin_files = [f for f in os.listdir(input_dir) if f.endswith('.bin')]

    for file_name in tqdm(bin_files, desc="Converting .bin to .pcd"):
        bin_path = os.path.join(input_dir, file_name)
        pcd_name = os.path.splitext(file_name)[0] + '.pcd'
        pcd_path = os.path.join(output_dir, pcd_name)
        bin_to_pcd(bin_path, pcd_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Directory containing .bin files")
    parser.add_argument("--output_dir", help="Directory to save .pcd files")
    args = parser.parse_args()

    batch_convert(args.input_dir, args.output_dir)