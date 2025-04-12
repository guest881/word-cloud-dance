# -*- coding: utf-8 -*-
"""
Created on 2025/4/11 17:12
@author: Mingzhe
"""
import xml.etree.ElementTree as ET
import csv

# 视频参数
total_frames = 2532
video_duration = 105  # 105秒 = 1分45秒
fps = total_frames / video_duration  # 计算帧率 ≈24.114fps

# 解析XML
tree = ET.parse('D:\program\download\芒种，一想到你我就…… (P1. 横屏版).cmt.txt')
root = tree.getroot()

# 提取弹幕数据
danmu_data = []
for d in root.findall('d'):
    p = d.get('p').split(',')
    timestamp = float(p[0])  # 获取时间戳（秒）
    text = d.text.strip()  # 获取弹幕文本

    # 计算对应帧位置 (四舍五入并限制范围)
    frame_pos = min(round(timestamp * fps), total_frames - 1)

    danmu_data.append({
        "time_sec": timestamp,
        "frame": frame_pos,
        "text": text
    })

# 导出CSV
with open('D:\program\download\芒种_frames.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["time_sec", "frame", "text"])
    writer.writeheader()
    writer.writerows(danmu_data)
