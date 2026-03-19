"""
目标特征库 — V2.0核心新功能
从照片集预提取多尺度目标特征（HSV直方图 + Hu矩 + 多尺度模板）
"""

import os
import json
import cv2
import numpy as np
from glob import glob
from config import (
    HSV_HIST_SIZE, HSV_RANGES, TEMPLATE_SCALES,
)


class TargetFeatureBank:
    """从照片集预提取多尺度目标特征"""

    def __init__(self):
        self.templates = []     # 每个元素: dict{image, histogram, hu_moments, size_level, multi_scale}

    def load_from_dir(self, photo_dir):
        """
        加载照片集目录，提取所有目标特征

        目录结构:
            target_photos/
            ├── far_01.jpg
            ├── mid_01.jpg
            ├── near_01.jpg
            └── target_info.json   # {"crops": {"far_01.jpg": [x1,y1,x2,y2], ...}}
        """
        info_path = os.path.join(photo_dir, 'target_info.json')
        if not os.path.exists(info_path):
            print(f'[特征库] 未找到 target_info.json，尝试全图模式')
            return self._load_full_images(photo_dir)

        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)

        crops_info = info.get('crops', {})
        loaded = 0

        for img_file in sorted(glob(os.path.join(photo_dir, '*.jpg'))):
            fname = os.path.basename(img_file)
            img = cv2.imread(img_file)
            if img is None:
                continue

            # 如果有标注框，裁剪目标区域；否则用全图
            if fname in crops_info:
                x1, y1, x2, y2 = crops_info[fname]
                crop = img[y1:y2, x1:x2]
            else:
                crop = img

            if crop.size == 0:
                continue

            template = self._extract_features(crop, fname)
            self.templates.append(template)
            loaded += 1

        print(f'[特征库] 已加载 {loaded} 个目标模板')
        return loaded

    def _load_full_images(self, photo_dir):
        """无标注信息时，用全图作为模板"""
        loaded = 0
        for img_file in sorted(glob(os.path.join(photo_dir, '*.jpg'))):
            img = cv2.imread(img_file)
            if img is None:
                continue
            fname = os.path.basename(img_file)
            template = self._extract_features(img, fname)
            self.templates.append(template)
            loaded += 1
        print(f'[特征库] 已加载 {loaded} 个全图模板')
        return loaded

    def _extract_features(self, crop, name=''):
        """从一张裁剪图提取全部特征"""
        # 1. HSV颜色直方图
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1], None, HSV_HIST_SIZE, HSV_RANGES
        )
        cv2.normalize(histogram, histogram)

        # 2. Hu矩（形状特征，尺度+旋转不变）
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        # 对数变换，压缩数值范围
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        # 3. 多尺度缩放版本（用于模板匹配）
        h, w = crop.shape[:2]
        multi_scale = {}
        for scale in TEMPLATE_SCALES:
            new_w = max(8, int(w * scale))
            new_h = max(8, int(h * scale))
            resized = cv2.resize(crop, (new_w, new_h))
            multi_scale[scale] = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 4. 判断尺寸级别
        area = h * w
        if area < 50 * 50:
            size_level = 'far'
        elif area < 150 * 150:
            size_level = 'mid'
        else:
            size_level = 'near'

        return {
            'name': name,
            'image': crop,
            'histogram': histogram,
            'hu_moments': hu_moments,
            'size_level': size_level,
            'multi_scale': multi_scale,
        }

    def get_histograms(self):
        """获取所有模板的直方图列表"""
        return [t['histogram'] for t in self.templates]

    def get_hu_moments(self):
        """获取所有模板的Hu矩列表"""
        return [t['hu_moments'] for t in self.templates]

    def is_loaded(self):
        return len(self.templates) > 0
