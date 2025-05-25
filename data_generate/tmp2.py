import os
import json
import cv2
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


def morphology_on_color_image(img, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    b, g, r = cv2.split(img)
    b_close = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
    g_close = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
    r_close = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)
    return cv2.merge([b_close, g_close, r_close])


def fuse_images(original, closed, alpha=0.3, beta=0.7):
    fused = cv2.addWeighted(original, alpha, closed, beta, 0)
    return fused

def highlight_fusion(img1, img2):
    return np.maximum(img1, img2)

def generate_mask_from_boxes(bev_img, annotations, x_range, y_range, resolution):
    mask = np.zeros((bev_img.shape[0], bev_img.shape[1]), dtype=np.uint8)
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
            img_pts = np.array(img_pts, dtype=np.int32)

            # 提取检测框内部非黑色区域作为 mask
            roi_mask = np.zeros_like(mask)
            cv2.fillPoly(roi_mask, [img_pts], 255)

            # 仅保留 roi 内部非黑像素
            gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
            non_black = gray > 10
            combined_mask = np.logical_and(roi_mask > 0, non_black)
            mask[combined_mask] = 255

        except Exception as e:
            print(f"[WARNING] Failed to process object: {e}")

    return mask

def draw_bev_with_boxes(bev_img, annotations, x_range, y_range, resolution):
    H, W, _ = bev_img.shape
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

def process_batch(bev_dir, json_dir, output_img_dir, output_mask_dir,
                  x_range=(0, 300), y_range=(-200, 200), resolution=0.1,
                  kernel_size=5, alpha=0.3, beta=0.7):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    files = [f for f in os.listdir(bev_dir) if f.endswith(".png")]

    for fname in tqdm(files, desc="Processing"):
        base_name = os.path.splitext(fname)[0]
        bev_path = os.path.join(bev_dir, fname)
        json_path = os.path.join(json_dir, base_name + ".json")

        if not os.path.exists(json_path):
            print(f"[WARNING] No annotation for {fname}, skipping.")
            continue

        bev_img = cv2.imread(bev_path)
        if bev_img is None:
            print(f"[ERROR] Failed to load image: {fname}")
            continue

        # Step 1: 闭运算增强
        closed = morphology_on_color_image(bev_img, kernel_size=kernel_size)
        # fused = fuse_images(bev_img, closed, alpha=alpha, beta=beta)
        fused = highlight_fusion(bev_img, closed)

        # Step 2: 加载标注信息
        try:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load json {json_path}: {e}")
            continue

        # Step 3: 绘制检测框
        fused_with_boxes = fused.copy()
        fused_with_boxes = draw_bev_with_boxes(fused_with_boxes, annotations, x_range, y_range, resolution)

        # Step 4: 生成 mask（只包含Car内非黑区域）
        mask = generate_mask_from_boxes(fused, annotations, x_range, y_range, resolution)

        # Step 5: 保存图像与 mask
        cv2.imwrite(os.path.join(output_img_dir, fname), fused_with_boxes)
        cv2.imwrite(os.path.join(output_mask_dir, base_name + "_mask.png"), mask)


if __name__ == "__main__":
    bev_input_dir = "./data_test/bev"
    json_input_dir = "./data_test/annotations"
    output_img_dir = "./data_test/bev_processed"
    output_mask_dir = "./data_test/bev_masks"

    process_batch(
        bev_dir=bev_input_dir,
        json_dir=json_input_dir,
        output_img_dir=output_img_dir,
        output_mask_dir=output_mask_dir,
        x_range=(0, 300),
        y_range=(-200, 200),
        resolution=0.1,
        kernel_size=5,
        alpha=0.3,
        beta=0.7
    )