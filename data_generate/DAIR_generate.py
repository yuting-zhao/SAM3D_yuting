import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

class BEVGenerator:
    def __init__(self, resolution=0.1, x_range=(0, 300), y_range=(-200, 200), z_range=(None, None), remove_ground_points=False):
        self.res = resolution
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.H = int((self.x_max - self.x_min) / self.res)
        self.W = int((self.y_max - self.y_min) / self.res)
        self.remove_ground_points = remove_ground_points

    def init_bev(self, B, channel=3, fill_value=0, device="cpu"):
        return torch.full((B, self.H, self.W, channel), fill_value, dtype=torch.float32, device=device)

    def ranks_from_coors(self, pcs):
        x = pcs[:, :, 0]
        y = pcs[:, :, 1]
        rank_x = ((x - self.x_min) / self.res).long().clamp(0, self.H - 1)
        rank_y = ((y - self.y_min) / self.res).long().clamp(0, self.W - 1)
        return rank_x, rank_y

    def distinguish_ground(self, point_clouds: torch.Tensor):
        ground_mask = []
        if len(point_clouds.shape) == 2:
            point_clouds = point_clouds.unsqueeze(0)

        for batch_idx in range(point_clouds.shape[0]):
            cur_pc = point_clouds[batch_idx]  # N x 4
            height = (cur_pc[:, 2] * 100).to(torch.int32)  # 转换为 cm 精度
            height, _ = torch.sort(height)
            lowest = height[0]
            hist = torch.bincount(height - lowest)
            max_bin = hist.argmax()
            max_num_height = max_bin * 0.01 + lowest * 0.01
            cur_ground_mask = cur_pc[:, 2] < max_num_height + 0.2  # +0.2m 作为缓冲
            ground_mask.append(cur_ground_mask)

        ground_mask = torch.stack(ground_mask)  # B x N
        return ground_mask

    def get_height_intensity_rgb_bev(self, point_clouds):
        B, N, _ = point_clouds.shape
        pc_z = point_clouds[:, :, 2]
        pc_i = point_clouds[:, :, 3]

        rank_x, rank_y = self.ranks_from_coors(point_clouds)
        batch_id = torch.arange(B).view(B, 1).expand(B, N)

        bev_map = self.init_bev(B, 3, 0, point_clouds.device)

        if self.remove_ground_points:
            ground_mask = self.distinguish_ground(point_clouds)

        # 归一化 Z（非线性，强调地面+2m 高度变化）
        if self.remove_ground_points:
            ground_mask = self.distinguish_ground(point_clouds)
            ground_height = pc_z.masked_select(ground_mask).view(B, -1).mean(dim=1, keepdim=True)  # B x 1
            relative_z = pc_z - ground_height  # B x N
            center = 2.0  # 地面+2m
            width = 2.0
            z_norm = torch.exp(-((relative_z - center) ** 2) / (2 * width ** 2))
        else:
            z_min, z_max = pc_z.min(), pc_z.max()
            z_norm = (pc_z - z_min) / (z_max - z_min + 1e-6)

        # 归一化强度
        pc_i = pc_i.clamp(min=1e-3).log()
        i_min, i_max = pc_i.min(), pc_i.max()
        i_norm = (pc_i - i_min) / (i_max - i_min + 1e-6)

        z_np = z_norm.view(B * N).cpu().numpy()
        color_map = plt.get_cmap('jet')
        base_colors = torch.from_numpy(color_map(z_np)[:, :3] * 255).to(torch.float32)
        base_colors = base_colors.view(B, N, 3).to(point_clouds.device)

        i_norm = i_norm.unsqueeze(-1)
        final_colors = base_colors * i_norm
        final_colors = final_colors.clamp(0, 255)
        
        # 添加这一行实现反色
        final_colors = 255 - final_colors

        if self.remove_ground_points:
            final_colors.masked_fill_(ground_mask.unsqueeze(-1), 0)

        bev_map[batch_id, rank_x, rank_y, :] = final_colors

        # # 替换背景像素（颜色值非常小的）为 ImageNet 风格的背景颜色
        # background_color = torch.tensor([123, 116, 103], dtype=torch.float32, device=point_clouds.device)

        # # 条件：RGB 所有通道都小于 10（表示为背景）
        # mask_background = (bev_map < 10).all(dim=-1)  # shape: (B, H, W)
        # bev_map[mask_background] = background_color
        
        return bev_map


def pcd_to_tensor(pcd_path):
    pcd = o3d.t.io.read_point_cloud(pcd_path)  # 用 t.io 支持 intensity 字段读取
    points = pcd.point["positions"].numpy()
    if "intensity" in pcd.point:
        intensities = pcd.point["intensity"].numpy()
    else:
        intensities = np.ones((points.shape[0], 1)) * 0.5

    if intensities.ndim == 1:
        intensities = intensities[:, None]

    point_cloud = np.hstack([points, intensities])
    tensor = torch.from_numpy(point_cloud).unsqueeze(0).float()
    return tensor


def batch_generate_bev(input_dir, output_dir, max_files=None):
    os.makedirs(output_dir, exist_ok=True)
    bev_gen = BEVGenerator(
        resolution=0.1,
        x_range=(0, 150),
        y_range=(-60, 60),
        z_range=(-30,80),
        remove_ground_points=True
    )

    pcd_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pcd")])
    
    if max_files is not None:
        pcd_files = pcd_files[:max_files]
    
    for fname in tqdm(pcd_files):
        path = os.path.join(input_dir, fname)
        try:
            pc_tensor = pcd_to_tensor(path)
            bev = bev_gen.get_height_intensity_rgb_bev(pc_tensor)[0].cpu().numpy().astype(np.uint8)
            out_img = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)
            # out_img = bev
            cv2.imwrite(os.path.join(output_dir, fname.replace(".pcd", ".png")), out_img)
        except Exception as e:
            print(f"Error processing {fname}: {e}")


if __name__ == "__main__":
    batch_generate_bev("/home/zhaoyuting/DAIR/single-infrastructure-side-velodyne", "/home/zhaoyuting/DAIR/ImageNet_background_and_distance_improve/bev", max_files=1500)  # 可自定义高度范围