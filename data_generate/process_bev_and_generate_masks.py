import os
import cv2
import json
import numpy as np
from tqdm import tqdm


def get_box_corners(x, y, w, l, yaw):
    dx = l / 2
    dy = w / 2
    corners = np.array([
        [-dx, -dy],
        [-dx, dy],
        [dx, dy],
        [dx, -dy]
    ])
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    rotated = (rotation @ corners.T).T
    return rotated + np.array([x, y])


def draw_bev_with_boxes(bev_img, annotations, x_range, y_range, resolution):
    canvas = bev_img.copy()
    type_color = {
        'Car': (0, 0, 255),
        'Cyclist': (255, 0, 0),
        'Pedestrian': (0, 255, 0),
        'Trafficcone': (0, 165, 255)
    }

    for obj in annotations:
        obj_type = obj.get("type", "")
        if obj_type not in type_color:
            continue

        try:
            loc = obj["3d_location"]
            dims = obj["3d_dimensions"]
            rot = float(obj["rotation"])

            x = float(loc["x"])
            y = float(loc["y"])
            w = float(dims["w"])
            l = float(dims["l"])

            corners = get_box_corners(x, y, w, l, rot)
            img_pts = []
            for corner in corners:
                px = int((corner[1] - y_range[0]) / resolution)
                py = int((corner[0] - x_range[0]) / resolution)
                img_pts.append([px, py])
            img_pts = np.array(img_pts, dtype=np.int32)

            cv2.polylines(canvas, [img_pts], isClosed=True, color=type_color[obj_type], thickness=2)

            center_px = int((y - y_range[0]) / resolution)
            center_py = int((x - x_range[0]) / resolution)
            cv2.circle(canvas, (center_px, center_py), 2, type_color[obj_type], -1)
            cv2.putText(canvas, obj_type, (center_px, center_py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, type_color[obj_type], 1, cv2.LINE_AA)
        except Exception as e:
            print(f"[WARNING] Skipped object due to error: {e}")

    return canvas


# def morphology_on_color_image(img, kernel_size=5):
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     b, g, r = cv2.split(img)
#     b_new = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
#     g_new = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
#     r_new = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)
#     result = cv2.merge([b_new, g_new, r_new])
#     return result

def distance_aware_morphology(img, x_range, y_range, resolution,
                              dist_thresholds=(30, 60, 90),
                              kernel_sizes=(3, 5, 9, 12)):
    """
    根据距离分区域对图像做不同核大小的形态学闭操作。

    Parameters:
    - img: 输入的RGB图像（H×W×3）
    - x_range, y_range: BEV图像的空间范围
    - resolution: 每个像素对应的米数
    - dist_thresholds: 3个距离分界值 (e.g. 30, 60, 90)
    - kernel_sizes: 4个对应的 kernel size (e.g. 3, 5, 9, 12)
    """
    H, W, _ = img.shape

    # 计算每个像素在BEV空间的真实坐标
    xv, yv = np.meshgrid(
        np.linspace(y_range[0], y_range[1], W),
        np.linspace(x_range[0], x_range[1], H)
    )
    dist_map = np.sqrt(xv ** 2 + yv ** 2)

    # 创建四个距离区域掩码
    mask_1 = dist_map < dist_thresholds[0]                     # 0–30m
    mask_2 = (dist_map >= dist_thresholds[0]) & (dist_map < dist_thresholds[1])  # 30–60m
    mask_3 = (dist_map >= dist_thresholds[1]) & (dist_map < dist_thresholds[2])  # 60–90m
    mask_4 = dist_map >= dist_thresholds[2]                    # 90m+

    result = np.zeros_like(img)

    # 对每个通道分别处理
    for c in range(3):
        ch = img[:, :, c]
        out = np.zeros_like(ch)

        for mask, ksize in zip([mask_1, mask_2, mask_3, mask_4], kernel_sizes):
            if np.any(mask):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                processed = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kernel)
                out[mask] = processed[mask]

        result[:, :, c] = out

    return result


def highlight_fusion(original, closed):
    return np.maximum(original, closed)


def generate_mask_from_boxes(bev_img, annotations, x_range, y_range, resolution):
    H, W, _ = bev_img.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for obj in annotations:
        if obj.get("type", "") != "Car":
            continue

        try:
            loc = obj["3d_location"]
            dims = obj["3d_dimensions"]
            rot = float(obj["rotation"])

            x = float(loc["x"])
            y = float(loc["y"])
            w = float(dims["w"])
            l = float(dims["l"])

            corners = get_box_corners(x, y, w, l, rot)

            img_pts = []
            for corner in corners:
                px = int((corner[1] - y_range[0]) / resolution)
                py = int((corner[0] - x_range[0]) / resolution)
                img_pts.append([px, py])
            img_pts = np.array([img_pts], dtype=np.int32)

            mask_polygon = np.zeros_like(mask)
            cv2.fillPoly(mask_polygon, img_pts, 255)

            gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
            non_black = (gray > 10).astype(np.uint8)
            combined_mask = cv2.bitwise_and(mask_polygon, mask_polygon, mask=non_black)
            mask = cv2.bitwise_or(mask, combined_mask)
        except Exception as e:
            print(f"[WARNING] Skipped mask generation due to error: {e}")
            continue

    return mask


def process_batch(bev_dir, json_dir, output_dir_fused, output_dir_fused_with_boxes, output_dir_mask,
                  x_range=(0, 300), y_range=(-200, 200), resolution=0.1, kernel_size=5):
    os.makedirs(output_dir_fused, exist_ok=True)
    os.makedirs(output_dir_fused_with_boxes, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)

    bev_files = [f for f in os.listdir(bev_dir) if f.endswith(".png")]

    for fname in tqdm(bev_files, desc="Processing BEV"):
        base_name = os.path.splitext(fname)[0]
        bev_path = os.path.join(bev_dir, fname)
        json_path = os.path.join(json_dir, base_name + ".json")

        bev_img = cv2.imread(bev_path)
        if bev_img is None:
            print(f"[ERROR] Failed to load image: {bev_path}")
            continue

        closed_img = distance_aware_morphology(
                bev_img,
                x_range=x_range,
                y_range=y_range,
                resolution=resolution,
                dist_thresholds=(30, 60, 90),
                kernel_sizes=(3, 5, 8, 10)
            )
        fused_img = highlight_fusion(bev_img, closed_img)

        fused_save_path = os.path.join(output_dir_fused, fname)
        cv2.imwrite(fused_save_path, fused_img)

        if not os.path.exists(json_path):
            print(f"[WARNING] Annotation not found for {fname}, skipping boxes and mask.")
            continue

        try:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load json: {json_path}, error: {e}")
            continue

        fused_with_boxes = draw_bev_with_boxes(fused_img.copy(), annotations, x_range, y_range, resolution)
        fused_box_path = os.path.join(output_dir_fused_with_boxes, fname)
        cv2.imwrite(fused_box_path, fused_with_boxes)

        mask = generate_mask_from_boxes(fused_img, annotations, x_range, y_range, resolution)
        mask_path = os.path.join(output_dir_mask, base_name + "_mask.png")
        cv2.imwrite(mask_path, mask)


if __name__ == "__main__":
    # bev_input_dir = "./data_test/bev"
    # json_input_dir = "./data_test/annotations"

    # output_dir_fused = "./data_test/bev_fused_only"               # 融合后（无框）图像输出
    # output_dir_fused_with_boxes = "./data_test/bev_fused_with_boxes"  # 融合后绘制检测框图像输出
    # output_dir_mask = "./data_test/bev_masks"                         # 蒙版输出

    bev_input_dir = "/home/zhaoyuting/DAIR/ImageNet_background_and_distance_improve/bev"
    json_input_dir = "/home/zhaoyuting/DAIR/single-infrastructure-side/label/virtuallidar"

    output_dir_fused = "/home/zhaoyuting/DAIR/ImageNet_background_and_distance_improve/bev_fused_only"               # 融合后（无框）图像输出
    output_dir_fused_with_boxes = "/home/zhaoyuting/DAIR/ImageNet_background_and_distance_improve/bev_fused_with_boxes"  # 融合后绘制检测框图像输出
    output_dir_mask = "/home/zhaoyuting/DAIR/ImageNet_background_and_distance_improve/bev_masks"                         # 蒙版输出
    
    process_batch(
        bev_dir=bev_input_dir,
        json_dir=json_input_dir,
        output_dir_fused=output_dir_fused,
        output_dir_fused_with_boxes=output_dir_fused_with_boxes,
        output_dir_mask=output_dir_mask,
        x_range=(0, 150),
        y_range=(-60, 60),
        resolution=0.1,
        kernel_size=5
    )
