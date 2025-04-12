# -*- coding: utf-8 -*-
"""
Created on 2025/4/11 20:20
@author: Mingzhe
"""
"""
生成人像轮廓词云视频帧（增强复用+密度优化版）
"""
from wordcloud import WordCloud
from PIL import Image, ImageFilter
import cv2
import pandas as pd
import os
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import jieba
import numpy as np
from collections import deque

# ----------------------- 配置参数（新增复用队列参数）-----------------------
FONT_PATH = r'C:\Windows\Fonts\msyh.ttc'
HUMAN_DIR = r"D:\program\download\video_frames3"
OUTPUT_DIR = r'D:\program\download\video_frames4'
FRAME_SIZE = (1920, 1080)
COLOR_PALETTES = [
    ['#FF6B6B', '#4ECDC4', '#45B7D1'],  # 红蓝渐变
    ['#9B59B6', '#2ECC71', '#27AE60'],  # 紫绿渐变
    ['#E74C3C', '#F1C40F', '#F39C12'],  # 红黄渐变
    ['#3498DB', '#9B59B6', '#8E44AD'],   # 蓝紫渐变
    ['#1ABC9C', '#2ECC71', '#16A085']   # 青绿渐变
]
TOTAL_FRAMES = 2532
REUSE_WINDOW = 5  # 历史帧复用窗口大小

# ----------------------- 词云核心配置（密度优化参数）-----------------------
WC_PARAMS = {
    'font_path': FONT_PATH,
    'background_color': None,
    'mode': 'RGBA',
    'prefer_horizontal': 0.6,  # 降低横向偏好
    'relative_scaling': 0.8,  # 增强大小差异
    'collocations': False,
    'max_words': 25,  # 增加最大词汇量
    'width': FRAME_SIZE[0],
    'height': FRAME_SIZE[1],
    'min_font_size': 2,  # 允许更小字号
    'repeat': True  # 允许重复高频词
}


# ----------------------- 资源加载类（增强遮罩处理）-----------------------
class ResourceLoader:
    def __init__(self):
        self.df = pd.read_csv('D:\program\download\芒种_frames.csv')
        self.texts = self.df.set_index('frame')['text'].to_dict()
        self.mask_cache = {}  # 新增遮罩缓存

    def get_clean_text(self, frame):
        raw_text = self.texts.get(frame, '').strip()
        return ' '.join(jieba.cut(raw_text)) if raw_text else None

    def get_mask(self, frame):
        """ 优化遮罩预处理流程 """
        if frame in self.mask_cache:
            return self.mask_cache[frame]

        img_path = os.path.join(HUMAN_DIR, f'frame_{frame:04d}.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(FRAME_SIZE[::-1], dtype=np.uint8)  # 返回空遮罩

        # 新增形态学操作（提升区域连贯性）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        _, mask = cv2.threshold(processed.astype(np.uint8), 240, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)  # 关键修改点

        mask = cv2.medianBlur(mask, 3)  # 中值滤波去噪

        self.mask_cache[frame] = mask.astype(np.uint8)
        return self.mask_cache[frame]


# ----------------------- 词云生成与合成（增强复用逻辑）-----------------------
def create_wordcloud(text, mask, color_map, last_valid=None):
    if text:
        try:
            wc = WordCloud(mask=mask, colormap=color_map, **WC_PARAMS)
            return wc.generate(text).to_image()
        except:
            return last_valid  # 生成失败时回退
    return last_valid  # 直接复用历史数据


def process_frame(frame, loader, history_queue):
    # 1. 获取数据
    text = loader.get_clean_text(frame)
    mask = loader.get_mask(frame)
    color_idx = frame % len(COLOR_PALETTES)
    cmap = LinearSegmentedColormap.from_list(f'palette{color_idx}', COLOR_PALETTES[color_idx])

    # 2. 生成词云（优先使用最近5帧的有效词云）
    current_cloud = None
    reuse_attempts = list(history_queue)[::-1]  # 逆序尝试复用
    for prev_cloud in reuse_attempts:
        if prev_cloud is not None:
            current_cloud = prev_cloud
            break

    current_cloud = create_wordcloud(text, mask, cmap, current_cloud)

    # 3. 更新历史队列
    if current_cloud:
        history_queue.append(current_cloud)
    else:
        history_queue.append(history_queue[-1] if history_queue else None)

    # 4. 图层合成（优化合成逻辑）
    bg_layer = Image.new('RGB', FRAME_SIZE, (0, 0, 0))

    if current_cloud:
        # 修复关键点：确保像素值为整数
        human_mask = Image.fromarray(mask, mode='L').filter(ImageFilter.GaussianBlur(1))
        human_layer = human_mask.convert('RGBA')
        # 显式转换为整数像素值
        human_data = [(0, 0, 0, 255 - int(min(p * 1.2, 255))) for p in human_mask.getdata()]
        human_layer.putdata(human_data)

        combined = Image.alpha_composite(human_layer, current_cloud)
        final_image = bg_layer.copy()
        final_image.paste(combined, (0, 0), combined)
    else:
        final_image = bg_layer  # 完全无数据时返回黑帧

    # 5. 保存结果
    final_image.save(os.path.join(OUTPUT_DIR, f'frame_{frame:04d}.png'))
    return history_queue


# ----------------------- 主流程（新增历史队列）-----------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loader = ResourceLoader()
    history_queue = deque(maxlen=REUSE_WINDOW)  # 维护复用窗口

    for frame in tqdm(range(TOTAL_FRAMES), desc="生成词云帧", unit="frame"):
        try:
            history_queue = process_frame(frame, loader, history_queue)
        except Exception as e:
            print(f"帧{frame}处理错误: {str(e)}")
            # 插入空白帧保持队列连续
            history_queue.append(history_queue[-1] if history_queue else None)

    print("\n生成完成！所有帧保存在", OUTPUT_DIR)