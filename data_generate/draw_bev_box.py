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


def batch_draw_boxes(bev_dir, json_dir, output_dir,
                     x_range=(0, 300), y_range=(-200, 200), resolution=0.1):
    os.makedirs(output_dir, exist_ok=True)
    bev_files = [f for f in os.listdir(bev_dir) if f.endswith(".png")]

    for fname in tqdm(bev_files, desc="Processing BEV images"):
        base_name = os.path.splitext(fname)[0]
        bev_path = os.path.join(bev_dir, fname)
        json_path = os.path.join(json_dir, base_name + ".json")
        output_path = os.path.join(output_dir, fname)

        if not os.path.exists(json_path):
            print(f"[WARNING] Annotation not found for {fname}, skipping.")
            continue

        bev_img = cv2.imread(bev_path)
        if bev_img is None:
            print(f"[ERROR] Failed to load image: {bev_path}")
            continue

        try:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load json: {json_path}, error: {e}")
            continue

        result = draw_bev_with_boxes(bev_img, annotations, x_range, y_range, resolution)
        cv2.imwrite(output_path, result)


if __name__ == "__main__":
    # 修改为你自己的路径
    bev_input_dir = "./data_test/bev"
    json_input_dir = "./data_test/annotations"
    output_dir = "./data_test/bev_with_boxes"

    batch_draw_boxes(
        bev_dir=bev_input_dir,
        json_dir=json_input_dir,
        output_dir=output_dir,
        x_range=(0, 300),
        y_range=(-200, 200),
        resolution=0.1
    )