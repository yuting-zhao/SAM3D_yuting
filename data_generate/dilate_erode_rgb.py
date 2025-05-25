import os
import cv2
import numpy as np
from tqdm import tqdm

def morphology_on_color_image(img, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    b, g, r = cv2.split(img)
    b_new = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
    g_new = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
    r_new = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)
    return cv2.merge([b_new, g_new, r_new])

def fuse_images(original, processed, alpha=0.7, beta=0.7, gamma=0):
    # 加权融合，允许 alpha + beta ≠ 1
    return cv2.addWeighted(processed, alpha, original, beta, gamma)

def batch_process(input_dir, output_dir, kernel_size=5, alpha=0.7, beta=0.7):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])

    for fname in tqdm(files, desc="Processing"):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {fname}")
            continue

        closed = morphology_on_color_image(img, kernel_size=kernel_size)
        fused = fuse_images(img, closed, alpha=alpha, beta=beta)
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, fused)

if __name__ == "__main__":
    input_folder = "./data_test/bev"                # 原始 BEV 图像文件夹
    output_folder = "./data_test/bev_fused"         # 输出文件夹
    kernel_size = 5
    alpha = 0.7  # 闭运算图像的权重
    beta = 0.7   # 原图的权重

    batch_process(input_folder, output_folder, kernel_size, alpha, beta)