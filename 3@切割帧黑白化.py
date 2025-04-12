import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor


def process_image(input_path, output_path):
    # 读取图像并保留原始Alpha通道（如有）
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # 分离通道处理（兼容带透明度的PNG）
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        img_bgr = cv2.merge([b, g, r])
    else:
        img_bgr = img
        a = None

    # 定义白色范围（HSV空间更鲁棒）
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 240])  # 亮度>=240视为背景
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 形态学优化（消除噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 生成结果（背景黑，主体白）
    result = np.full_like(img_bgr, 255)  # 全白画布
    result[mask == 255] = 0  # 背景设为黑色

    # 保留原始透明度或生成新Alpha通道
    if a is not None:
        result = cv2.merge([result[..., 0], result[..., 1], result[..., 2], a])

    cv2.imwrite(output_path, result)


def batch_process(input_dir, output_dir, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file in files:
            in_path = os.path.join(input_dir, file)
            out_path = os.path.join(output_dir, file)
            executor.submit(process_image, in_path, out_path)


# 使用示例
batch_process(r"D:\program\download\video_frames2", r"D:\program\download\video_frames3")