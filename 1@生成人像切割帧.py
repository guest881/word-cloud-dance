import cv2
import os

# 配置参数
video_path = r"D:\program\download\video_heng_v2.avi"#人像切割后的视频
output_dir = r"D:\program\download\video_frames2"
target_size = None  # 设置为None则不调整尺寸

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 调整像素尺寸
    if target_size:
        frame = cv2.resize(frame, target_size)

    # 保存到指定路径
    save_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(save_path, frame)
    frame_count += 1
    print(frame_count)
cap.release()
print(f"共保存 {frame_count} 帧到 {output_dir}")