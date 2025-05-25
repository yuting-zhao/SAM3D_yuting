import os
import cv2
import numpy as np
from tqdm import tqdm

def convert_and_morph(input_dir, gray_dir, closed_dir, kernel_size=5):
    os.makedirs(gray_dir, exist_ok=True)
    os.makedirs(closed_dir, exist_ok=True)

    bev_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for fname in tqdm(bev_files, desc="Processing BEV images"):
        img_path = os.path.join(input_dir, fname)

        # 读取彩色 BEV 图
        bev_img = cv2.imread(img_path)

        # 转为灰度图
        gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
        gray_path = os.path.join(gray_dir, fname)
        cv2.imwrite(gray_path, gray)

        # 对灰度图做闭运算（先膨胀再腐蚀）
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        closed_path = os.path.join(closed_dir, fname)
        cv2.imwrite(closed_path, closed)

if __name__ == "__main__":
    convert_and_morph(
        input_dir="./data_test/bev",               # 输入：原始 BEV 彩色图像目录
        gray_dir="./data_test/bev_gray",           # 输出：灰度图目录
        closed_dir="./data_test/bev_closed",       # 输出：闭运算后图像目录
        kernel_size=5                              # 卷积核大小（可调节）
    )