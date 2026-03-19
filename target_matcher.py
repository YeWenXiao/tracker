"""
多特征融合匹配器 — 颜色+形状+YOLO综合判断"是不是那个目标"
"""

import cv2
import numpy as np
from config import (
    WEIGHT_COLOR, WEIGHT_SHAPE, WEIGHT_YOLO,
    MATCH_THRESHOLD, VERIFY_THRESHOLD,
    HSV_HIST_SIZE, HSV_RANGES, TEMPLATE_SCALES,
)

# ORB特征检测器(全局复用)
_orb = cv2.ORB_create(nfeatures=200)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


class TargetMatcher:
    """多特征融合目标匹配"""

    def __init__(self, feature_bank):
        self.bank = feature_bank
        self._hsv_lower = None
        self._hsv_upper = None
        if self.bank.is_loaded():
            self._compute_hsv_range()

    def match_crop(self, crop, yolo_conf=0.0):
        """
        对一个候选裁剪区域，与特征库中所有模板比对，返回最高综合分

        Args:
            crop: BGR图像裁剪
            yolo_conf: YOLO检测置信度 (0-1)

        Returns:
            (score, best_template_idx) 或 (0.0, -1) 如果不匹配
        """
        if crop is None or crop.size == 0 or not self.bank.is_loaded():
            return 0.0, -1

        # 裁剪太小没有区分度，直接跳过
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return 0.0, -1

        # 提取候选目标的特征（只用颜色+纹理，Hu矩区分度差已去掉）
        color_score = self._best_color_match(crop)
        texture_score = self._best_orb_match(crop)

        # 综合评分: 颜色0.45 + 纹理0.40 + YOLO0.15
        score = (0.45 * color_score +
                 0.40 * texture_score +
                 0.15 * yolo_conf)

        return score, 0

    def match_detections(self, frame, detections, threshold=None):
        """
        对YOLO检测到的所有候选目标，找出最匹配照片集的那个

        Args:
            frame: 完整帧
            detections: [(x1,y1,x2,y2,conf,cls), ...]
            threshold: 匹配阈值，默认用 MATCH_THRESHOLD

        Returns:
            (best_det, score) 或 (None, 0.0)
        """
        if threshold is None:
            threshold = MATCH_THRESHOLD

        best_det = None
        best_score = 0.0

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            score, _ = self.match_crop(crop, yolo_conf=conf)
            if score > best_score:
                best_score = score
                best_det = det

        if best_score >= threshold:
            return best_det, best_score
        return None, 0.0

    def verify_target(self, crop, yolo_conf=0.0):
        """
        追踪中持续验证：当前追踪的还是原目标吗？
        验证时用颜色分数单独判断，ORB对实时裁剪不稳定

        Returns:
            True 如果仍然匹配，False 如果可能跟错了
        """
        if crop is None or crop.size == 0 or not self.bank.is_loaded():
            return False
        color_score = self._best_color_match(crop)
        # 颜色匹配>0.3就认为没跟错，避免ORB波动导致误判
        return color_score >= VERIFY_THRESHOLD

    def color_search(self, frame):
        """
        用目标颜色的HSV范围在全画面搜索 — 不依赖YOLO

        Returns:
            [(x1, y1, x2, y2, score), ...] 候选列表，按分数降序
        """
        if not self.bank.is_loaded():
            return []

        h, w = frame.shape[:2]
        frame_area = h * w

        # 降采样加速（半分辨率搜索）
        small = cv2.resize(frame, (w // 2, h // 2))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # 延迟初始化（照片集可能在构造后才加载）
        if self._hsv_lower is None:
            self._compute_hsv_range()

        # HSV范围掩码
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # 形态学
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # 半分辨率下太小
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # 映射回原分辨率
            x, y, bw, bh = x * 2, y * 2, bw * 2, bh * 2
            real_area = bw * bh

            # 大小过滤: 目标不可能占画面5%以上(1x zoom下约1-2%)
            if real_area > frame_area * 0.05:
                continue
            # 也不能太小
            if bw < 15 or bh < 15:
                continue
            # 宽高比过滤
            ratio = max(bw, bh) / max(min(bw, bh), 1)
            if ratio > 4:
                continue

            # 扩大裁剪区域
            pad_x = int(bw * 0.1)
            pad_y = int(bh * 0.1)
            cx1 = max(0, x - pad_x)
            cy1 = max(0, y - pad_y)
            cx2 = min(w, x + bw + pad_x)
            cy2 = min(h, y + bh + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            score, _ = self.match_crop(crop)
            candidates.append((cx1, cy1, cx2, cy2, score))

        # 按分数降序
        candidates.sort(key=lambda c: c[4], reverse=True)
        return candidates[:5]

    def _compute_hsv_range(self):
        """从模板中心区域计算目标HSV颜色范围（排除边缘背景）"""
        all_h, all_s, all_v = [], [], []
        for tmpl in self.bank.templates:
            crop = tmpl['image']
            ch, cw = crop.shape[:2]
            # 只取中心60%区域，避免边缘的背景像素
            margin_y = max(1, int(ch * 0.2))
            margin_x = max(1, int(cw * 0.2))
            center = crop[margin_y:ch-margin_y, margin_x:cw-margin_x]
            if center.size == 0:
                center = crop
            hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
            all_h.extend(hsv[:, :, 0].flatten().tolist())
            all_s.extend(hsv[:, :, 1].flatten().tolist())
            all_v.extend(hsv[:, :, 2].flatten().tolist())

        all_h = np.array(all_h)
        all_s = np.array(all_s)
        all_v = np.array(all_v)

        # 用25-75百分位确定核心范围，再加容差
        h_lo, h_hi = np.percentile(all_h, 25), np.percentile(all_h, 75)
        s_lo, s_hi = np.percentile(all_s, 25), np.percentile(all_s, 75)
        v_lo, v_hi = np.percentile(all_v, 25), np.percentile(all_v, 75)

        # 加固定容差
        h_margin = 12
        s_margin = 35
        v_margin = 45

        self._hsv_lower = np.array([max(0, h_lo - h_margin),
                                     max(50, s_lo - s_margin),   # S最低50，排除灰色/白色
                                     max(100, v_lo - v_margin)], # V最低100，排除暗区
                                    dtype=np.uint8)
        self._hsv_upper = np.array([min(180, h_hi + h_margin),
                                     min(255, s_hi + s_margin),
                                     min(255, v_hi + v_margin)], dtype=np.uint8)
        print(f'[颜色搜索] HSV范围: {self._hsv_lower} ~ {self._hsv_upper}')

    def template_search(self, frame):
        """
        无YOLO检测结果时，颜色搜索+模板匹配

        Returns:
            (x1, y1, x2, y2, score) 或 None
        """
        candidates = self.color_search(frame)
        if candidates and candidates[0][4] > MATCH_THRESHOLD:
            return candidates[0]
        return None

    def _best_color_match(self, crop):
        """与特征库中所有模板做颜色直方图比较，返回最高分(0-1)"""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, HSV_HIST_SIZE, HSV_RANGES)
        cv2.normalize(hist, hist)

        best = 0.0
        for tmpl_hist in self.bank.get_histograms():
            score = cv2.compareHist(hist, tmpl_hist, cv2.HISTCMP_CORREL)
            # CORREL范围-1~1，映射到0~1
            score = max(0.0, (score + 1.0) / 2.0)
            best = max(best, score)

        return best

    def _best_shape_match(self, crop):
        """与特征库中所有模板做Hu矩比较，返回最高分(0-1)"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        best = 0.0
        for tmpl_hu in self.bank.get_hu_moments():
            # 欧氏距离，加大衰减增强区分度
            dist = np.linalg.norm(hu - tmpl_hu)
            score = 1.0 / (1.0 + dist * 0.5)
            best = max(best, score)

        return best

    def _best_orb_match(self, crop):
        """ORB特征点匹配 — 纹理级区分，对非目标区分度高"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # 统一大小避免特征点数量差异
        gray = cv2.resize(gray, (128, 128))
        kp1, des1 = _orb.detectAndCompute(gray, None)

        if des1 is None or len(des1) < 5:
            return 0.0

        best = 0.0
        for tmpl in self.bank.templates:
            tmpl_gray = cv2.cvtColor(tmpl['image'], cv2.COLOR_BGR2GRAY)
            tmpl_gray = cv2.resize(tmpl_gray, (128, 128))
            kp2, des2 = _orb.detectAndCompute(tmpl_gray, None)

            if des2 is None or len(des2) < 5:
                continue

            matches = _bf.match(des1, des2)
            if not matches:
                continue

            # 取前20个最佳匹配的距离
            matches = sorted(matches, key=lambda m: m.distance)
            top_n = min(20, len(matches))
            avg_dist = sum(m.distance for m in matches[:top_n]) / top_n

            # 距离转相似度 (ORB距离范围0-256)
            score = max(0.0, 1.0 - avg_dist / 120.0)
            best = max(best, score)

        return best
