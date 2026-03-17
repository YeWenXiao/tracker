"""
目标识别测试 (多方法 + 分步计时)
方法:
  1. 多尺度模板匹配
  2. ORB 特征点匹配 + 单应性
  3. SIFT 特征点匹配 + 单应性
  4. HSV 颜色直方图反投影
  5. 边缘轮廓匹配

用法:
  python recognize.py --batch              # 批量测试
  python recognize.py --image xxx.jpg      # 单张测试
  python recognize.py                      # MIPI CSI 实时 (默认, 低延迟)
  python recognize.py --fast               # MIPI CSI 快速模式 (~30fps)
  python recognize.py --fast --save        # 快速模式 + 录像保存
  python recognize.py --rtsp               # RTSP 实时 (备用)
  python recognize.py --rtsp --fast        # RTSP 快速模式
  python recognize.py --sensor-id 1        # 指定 MIPI 摄像头ID
  python recognize.py --verbose            # DEBUG 级别日志
  python recognize.py --log-file run.log   # 日志写入文件
  python recognize.py --auto-track          # 云台自动跟踪目标
"""

import cv2
import numpy as np
import os
import json
import argparse
import time
import glob
import logging

from logger import setup_logger
from tracker import KalmanTracker

log = setup_logger("tracker")


def load_config(path="config.yaml"):
    """加载 YAML 配置文件，不存在则返回空 dict"""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # 没有 PyYAML 时用简单解析
        import re
        cfg = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    m = re.match(r"(\w[\w.]*\w?)\s*:\s*(.+)", line)
                    if m:
                        cfg[m.group(1)] = m.group(2).strip('"').strip()
        except FileNotFoundError:
            pass
        return cfg
    except FileNotFoundError:
        return {}


class TargetRecognizer:
    def __init__(self, targets_dir="targets"):
        self.targets = []
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)
        # FLANN for SIFT
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # CUDA 加速检测
        self.use_cuda = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.cuda_template_matcher = None
        if self.use_cuda:
            try:
                self.cuda_template_matcher = cv2.cuda.createTemplateMatching(
                    cv2.CV_8U, cv2.TM_CCOEFF_NORMED)
                log.info('CUDA GPU 加速已启用 (cvtColor/resize/templateMatch)')
            except Exception as e:
                log.info('CUDA GPU 加速已启用 (cvtColor/resize), 模板匹配CUDA不可用: %s', e)
        else:
            log.info('CUDA GPU 不可用, 使用 CPU 模式')
        self._load(targets_dir)

    def _load(self, targets_dir):
        info_path = os.path.join(targets_dir, "target_info.json")
        with open(info_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        for ann in annotations:
            img = cv2.imread(os.path.join(targets_dir, ann["crop"]))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # ORB
            orb_kp, orb_des = self.orb.detectAndCompute(gray, None)
            # SIFT
            sift_kp, sift_des = self.sift.detectAndCompute(gray, None)
            # HSV 直方图
            hist_hs = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist_hs, hist_hs)
            # 用于反投影的直方图
            hist_bp = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist_bp, hist_bp, 0, 255, cv2.NORM_MINMAX)
            # 边缘
            edges = cv2.Canny(gray, 50, 150)

            self.targets.append({
                "name": ann["crop"],
                "source": ann["source"],
                "image": img,
                "gray": gray,
                "hsv": hsv,
                "orb_kp": orb_kp, "orb_des": orb_des,
                "sift_kp": sift_kp, "sift_des": sift_des,
                "hist_hs": hist_hs,
                "hist_bp": hist_bp,
                "edges": edges,
            })
        log.info("已加载 %d 个目标模板", len(self.targets))

    def recognize(self, scene_bgr, fast=False):
        """
        在场景中识别目标
        fast=True: 降分辨率模板 + ORB + 颜色反投影 (~160ms, ~6fps)
        fast=False: 全部5种方法 (~1000ms, ~1fps)
        返回: results_list, timing_dict
        """
        scene_h, scene_w = scene_bgr.shape[:2]
        if self.use_cuda:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(scene_bgr)
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                scene_gray = gpu_gray.download()
                gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
                scene_hsv = gpu_hsv.download()
            except cv2.error as e:
                if 'out of memory' in str(e).lower() or 'insufficient' in str(e).lower():
                    log.warning('CUDA 显存不足，切换到 CPU 模式')
                    self.use_cuda = False
                    self.cuda_template_matcher = None
                    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
                    scene_hsv = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2HSV)
                else:
                    raise
        else:
            scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
            scene_hsv = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2HSV)
        timing = {}
        all_results = []

        aspect_ratios = [t["gray"].shape[0] / t["gray"].shape[1] for t in self.targets]
        avg_ar = np.mean(aspect_ratios)
        min_dim = int(min(scene_w, scene_h) * 0.03)

        # === 方法1: 多尺度模板匹配 ===
        t0 = time.time()
        template_results = []
        if fast:
            # 快速模式: 场景缩小到1/3，减少尺度数
            ds = 1.0 / 3
            if self.use_cuda:
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(scene_gray)
                gpu_small = cv2.cuda.resize(gpu_gray, (int(scene_gray.shape[1]*ds), int(scene_gray.shape[0]*ds)))
                scene_small = gpu_small.download()
            else:
                scene_small = cv2.resize(scene_gray, None, fx=ds, fy=ds)
            scales = np.arange(0.3, 1.5, 0.4)  # 3个尺度 vs 全量12个
        else:
            scene_small = scene_gray
            ds = 1.0
            scales = np.arange(0.3, 1.5, 0.1)
        # CUDA: 场景图上传一次，多模板复用
        gpu_scene_tmpl = None
        if self.cuda_template_matcher is not None:
            try:
                gpu_scene_tmpl = cv2.cuda_GpuMat()
                gpu_scene_tmpl.upload(scene_small)
            except Exception:
                gpu_scene_tmpl = None

        for t in self.targets:
            th, tw = t["gray"].shape[:2]
            for scale in scales:
                nw, nh = int(tw * scale * ds), int(th * scale * ds)
                if nw < max(min_dim * ds, 5) or nh < max(min_dim * ds, 5):
                    continue
                sw, sh = scene_small.shape[1], scene_small.shape[0]
                if nw > sw or nh > sh:
                    continue
                resized = cv2.resize(t["gray"], (nw, nh))
                # CUDA 模板匹配 (单通道 CV_8U)
                if gpu_scene_tmpl is not None:
                    try:
                        gpu_tmpl = cv2.cuda_GpuMat()
                        gpu_tmpl.upload(resized)
                        gpu_result = self.cuda_template_matcher.match(gpu_scene_tmpl, gpu_tmpl)
                        res = gpu_result.download()
                    except Exception:
                        res = cv2.matchTemplate(scene_small, resized, cv2.TM_CCOEFF_NORMED)
                else:
                    res = cv2.matchTemplate(scene_small, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.6:
                    # 坐标映射回原始分辨率
                    ox, oy = int(max_loc[0] / ds), int(max_loc[1] / ds)
                    ow, oh = int(nw / ds), int(nh / ds)
                    template_results.append((max_val, ox, oy, ow, oh, "template"))
        timing["template"] = time.time() - t0
        log.debug("模板匹配: %.0fms, %d 个候选", timing["template"]*1000, len(template_results))
        all_results.extend(template_results)

        # 快速模式: 模板高置信度命中时跳过后续方法
        high_conf = fast and any(s > 0.8 for s, *_ in template_results)

        # === 方法2: ORB 特征匹配 ===
        if not high_conf:
            t0 = time.time()
            orb_results = []
            orb_kp_s, orb_des_s = self.orb.detectAndCompute(scene_gray, None)
            if orb_des_s is not None:
                for t in self.targets:
                    if t["orb_des"] is None:
                        continue
                    matches = self.bf_hamming.knnMatch(t["orb_des"], orb_des_s, k=2)
                    good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.70 * n.distance]
                    if len(good) >= 10:
                        box = self._homography_box(t, good, orb_kp_s, "orb_kp",
                                                   scene_w, scene_h, min_dim, avg_ar)
                        if box:
                            orb_results.append(box)
            timing["orb"] = time.time() - t0
            log.debug("ORB匹配: %.0fms, %d 个候选", timing["orb"]*1000, len(orb_results))
            all_results.extend(orb_results)

        if not fast:
            # === 方法3: SIFT 特征匹配 (FLANN加速) ===
            t0 = time.time()
            sift_results = []
            sift_kp_s, sift_des_s = self.sift.detectAndCompute(scene_gray, None)
            if sift_des_s is not None and len(sift_des_s) >= 2:
                for t in self.targets:
                    if t["sift_des"] is None or len(t["sift_des"]) < 2:
                        continue
                    matches = self.flann.knnMatch(t["sift_des"], sift_des_s, k=2)
                    good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.70 * n.distance]
                    if len(good) >= 10:
                        box = self._homography_box(t, good, sift_kp_s, "sift_kp",
                                                   scene_w, scene_h, min_dim, avg_ar)
                        if box:
                            box = (box[0], box[1], box[2], box[3], box[4], "sift")
                            sift_results.append(box)
            timing["sift"] = time.time() - t0
            log.debug("SIFT匹配: %.0fms, %d 个候选", timing["sift"]*1000, len(sift_results))
            all_results.extend(sift_results)

        # === 方法4: HSV 颜色反投影 ===
        if not high_conf:
            t0 = time.time()
            bp_results = []
            for t in self.targets:
                backproj = cv2.calcBackProject([scene_hsv], [0, 1], t["hist_bp"],
                                               [0, 180, 0, 256], 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                backproj = cv2.morphologyEx(backproj, cv2.MORPH_CLOSE, kernel, iterations=2)
                backproj = cv2.morphologyEx(backproj, cv2.MORPH_OPEN, kernel, iterations=1)
                _, thresh = cv2.threshold(backproj, 80, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_dim * min_dim:
                        continue
                    if area > scene_w * scene_h * 0.5:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    det_ar = h / max(w, 1)
                    if det_ar < avg_ar * 0.3 or det_ar > avg_ar * 3.0:
                        continue
                    score = min(area / (scene_w * scene_h * 0.05), 1.0)
                    bp_results.append((score, x, y, w, h, "color_bp"))
            timing["color_bp"] = time.time() - t0
            log.debug("颜色反投影: %.0fms, %d 个候选", timing["color_bp"]*1000, len(bp_results))
            all_results.extend(bp_results)

        if not fast:
            # === 方法5: 边缘模板匹配 ===
            t0 = time.time()
            edge_results = []
            scene_edges = cv2.Canny(scene_gray, 50, 150)
            for t in self.targets:
                th, tw = t["edges"].shape[:2]
                for scale in np.arange(0.4, 1.4, 0.15):
                    nw, nh = int(tw * scale), int(th * scale)
                    if nw < min_dim or nh < min_dim:
                        continue
                    if nw > scene_w or nh > scene_h:
                        continue
                    resized = cv2.resize(t["edges"], (nw, nh))
                    res = cv2.matchTemplate(scene_edges, resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > 0.35:
                        edge_results.append((max_val, max_loc[0], max_loc[1], nw, nh, "edge"))
            timing["edge"] = time.time() - t0
            log.debug("边缘匹配: %.0fms, %d 个候选", timing["edge"]*1000, len(edge_results))
            all_results.extend(edge_results)

        # === NMS + 颜色验证 ===
        t0 = time.time()
        all_results = self._nms(all_results, 0.3)

        verified = []
        for score, x, y, w, h, method in all_results:
            x, y = max(0, x), max(0, y)
            x2 = min(x + w, scene_w)
            y2 = min(y + h, scene_h)
            if x2 - x < min_dim or y2 - y < min_dim:
                continue
            region = scene_bgr[y:y2, x:x2]
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            color_score = max(
                cv2.compareHist(t["hist_hs"], hist, cv2.HISTCMP_CORREL)
                for t in self.targets
            )
            combined = 0.4 * score + 0.6 * max(color_score, 0)
            if combined >= 0.45:
                verified.append((combined, x, y, x2-x, y2-y, method))

        verified.sort(key=lambda r: r[0], reverse=True)
        timing["nms_verify"] = time.time() - t0
        timing["total"] = sum(timing.values())
        log.debug("总计: %.0fms, %d 个结果", timing["total"]*1000, len(verified))

        return verified[:5], timing

    def _homography_box(self, t, good_matches, scene_kps, kp_key,
                        scene_w, scene_h, min_dim, avg_ar):
        """从好的匹配点计算单应性变换，返回检测框"""
        src_pts = np.float32([t[kp_key][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([scene_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return None
        inliers = mask.ravel().sum()
        if inliers < 8:
            return None
        h, w = t["gray"].shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, M)
        x, y, bw, bh = cv2.boundingRect(proj)
        if bw < min_dim or bh < min_dim:
            return None
        if bw > scene_w * 0.8 or bh > scene_h * 0.8:
            return None
        det_ar = bh / max(bw, 1)
        if det_ar < avg_ar * 0.3 or det_ar > avg_ar * 3.0:
            return None
        score = min(inliers / 20.0, 1.0)
        return (score, x, y, bw, bh, "orb")

    def _nms(self, results, thresh):
        if not results:
            return []
        results.sort(key=lambda r: r[0], reverse=True)
        keep = []
        for s1, x1, y1, w1, h1, m1 in results:
            suppressed = False
            for s2, x2, y2, w2, h2, m2 in keep:
                xa, ya = max(x1, x2), max(y1, y2)
                xb, yb = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
                inter = max(0, xb-xa) * max(0, yb-ya)
                union = w1*h1 + w2*h2 - inter
                if union > 0 and inter / union > thresh:
                    suppressed = True
                    break
            if not suppressed:
                keep.append((s1, x1, y1, w1, h1, m1))
        return keep


def get_system_stats():
    """获取系统状态 (Jetson 温度 + 内存)"""
    stats = {}
    # CPU温度 (Jetson)
    try:
        with open("/sys/devices/virtual/thermal/thermal_zone0/temp") as f:
            stats["cpu_temp"] = int(f.read().strip()) / 1000
    except Exception:
        pass
    # GPU温度 (Jetson)
    try:
        with open("/sys/devices/virtual/thermal/thermal_zone1/temp") as f:
            stats["gpu_temp"] = int(f.read().strip()) / 1000
    except Exception:
        pass
    # 内存使用
    try:
        import psutil
        mem = psutil.virtual_memory()
        stats["mem_percent"] = mem.percent
    except Exception:
        pass
    return stats


def draw_results(img, results, timing=None, label="", fps=None, latency_ms=None,
                 roi=None, system_stats=None):
    """在图上画识别结果和计时信息"""
    display = img.copy()
    for score, x, y, w, h, method in results:
        color = (0, 255, 0) if score > 0.6 else (0, 255, 255) if score > 0.4 else (0, 0, 255)
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
        cv2.putText(display, f"{method} {score:.2f}", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # ROI 蓝色虚线框
    if roi is not None:
        rx, ry, rw, rh = roi
        dash_len = 10
        gap_len = 6
        color_roi = (255, 150, 0)  # 蓝色
        for edge_pts in [
            ((rx, ry), (rx + rw, ry)),
            ((rx + rw, ry), (rx + rw, ry + rh)),
            ((rx + rw, ry + rh), (rx, ry + rh)),
            ((rx, ry + rh), (rx, ry)),
        ]:
            pt1, pt2 = edge_pts
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            dist = max(int(np.sqrt(dx*dx + dy*dy)), 1)
            for i in range(0, dist, dash_len + gap_len):
                s = i / dist
                e = min((i + dash_len) / dist, 1.0)
                sp = (int(pt1[0] + dx * s), int(pt1[1] + dy * s))
                ep = (int(pt1[0] + dx * e), int(pt1[1] + dy * e))
                cv2.line(display, sp, ep, color_roi, 2)
        cv2.putText(display, "ROI", (rx + 4, ry + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_roi, 1)

    # 左上角信息
    y_off = 25
    if label:
        cv2.putText(display, label, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_off += 25
    # 实时 FPS 和延迟显示
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(display, fps_text, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_off += 22
    if latency_ms is not None:
        lat_color = (0, 255, 0) if latency_ms < 50 else (0, 255, 255) if latency_ms < 100 else (0, 0, 255)
        lat_text = f"Latency: {latency_ms:.1f}ms"
        cv2.putText(display, lat_text, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, lat_color, 2)
        y_off += 22
    if timing:
        for key in ["template", "orb", "sift", "color_bp", "edge", "nms_verify", "total"]:
            if key in timing:
                prefix = ">> " if key == "total" else "   "
                text = f"{prefix}{key}: {timing[key]*1000:.0f}ms"
                cv2.putText(display, text, (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_off += 20
    # 右下角系统状态 OSD
    if system_stats:
        img_h, img_w = display.shape[:2]
        lines = []
        if "cpu_temp" in system_stats:
            t = system_stats["cpu_temp"]
            lines.append((f"CPU: {t:.0f}C", (0, 0, 255) if t > 80 else (0, 255, 255)))
        if "gpu_temp" in system_stats:
            t = system_stats["gpu_temp"]
            lines.append((f"GPU: {t:.0f}C", (0, 0, 255) if t > 80 else (0, 255, 255)))
        if "mem_percent" in system_stats:
            m = system_stats["mem_percent"]
            lines.append((f"MEM: {m:.0f}%", (0, 0, 255) if m > 90 else (0, 255, 255)))
        if lines:
            line_h = 20
            x_right = img_w - 10
            y_bottom = img_h - 10
            for i, (text, color) in enumerate(reversed(lines)):
                y_pos = y_bottom - i * line_h
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(display, text, (x_right - tw, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return display


def print_timing(timing):
    """打印分步耗时"""
    log.info("分步耗时:")
    for key in ["template", "orb", "sift", "color_bp", "edge", "nms_verify"]:
        if key in timing:
            log.info("  %12s: %6.0f ms", key, timing[key]*1000)
    log.info("  %12s: %6.0f ms", "总计", timing.get('total', 0)*1000)


class FPSCounter:
    """滑动窗口 FPS 计算器"""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []

    def tick(self):
        """记录一帧时间戳，返回当前 FPS"""
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


def main():
    parser = argparse.ArgumentParser(
        description="A8mini 目标识别追踪系统 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python recognize.py --fast            # MIPI 快速模式
  python recognize.py --rtsp            # RTSP 模式
  python recognize.py --auto-track      # 云台自动跟踪
  python recognize.py --fast --save     # 快速模式 + 录像
  python recognize.py --batch           # 批量测试
  python recognize.py --image x.jpg     # 单张测试
""")

    # 视频源参数组
    source_group = parser.add_argument_group("视频源")
    source_group.add_argument("--mipi", action="store_true",
                              help="使用 MIPI CSI 摄像头 (默认已启用)")
    source_group.add_argument("--rtsp", action="store_true",
                              help="使用 RTSP 视频源")
    source_group.add_argument("--rtsp-url", default="rtsp://192.168.144.25:8554/main.264",
                              help="RTSP 地址 (默认: %(default)s)")
    source_group.add_argument("--sensor-id", type=int, default=0,
                              help="MIPI 摄像头编号 (默认: %(default)s)")
    source_group.add_argument("--width", type=int, default=1280,
                              help="MIPI 采集宽度 (默认: %(default)s)")
    source_group.add_argument("--height", type=int, default=720,
                              help="MIPI 采集高度 (默认: %(default)s)")
    source_group.add_argument("--fps-cap", type=int, default=30,
                              help="MIPI 采集帧率 (默认: %(default)s)")

    # 识别参数组
    recog_group = parser.add_argument_group("识别")
    recog_group.add_argument("--image", type=str, help="单张测试图片路径")
    recog_group.add_argument("--batch", action="store_true",
                             help="批量测试 captures/ 下所有图片")
    recog_group.add_argument("--fast", action="store_true",
                             help="快速模式: 只用 ORB+颜色 (~20ms)")

    # 追踪参数组
    track_group = parser.add_argument_group("追踪")
    track_group.add_argument("--auto-track", action="store_true",
                             help="云台自动跟踪目标")

    # 输出参数组
    output_group = parser.add_argument_group("输出")
    output_group.add_argument("--save", action="store_true",
                              help="保存识别视频到 recordings/")
    output_group.add_argument("--verbose", action="store_true",
                              help="DEBUG 级别日志输出")
    output_group.add_argument("--log-file", type=str, default=None,
                              help="日志输出到文件")
    args = parser.parse_args()

    # 重新配置 logger (根据命令行参数)
    global log
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log = setup_logger("tracker", level=log_level, log_file=args.log_file)

    # 加载配置文件 (命令行参数优先)
    cfg = load_config()
    video_cfg = cfg.get("video", {})
    mipi_cfg = video_cfg.get("mipi", {})
    recog_cfg = cfg.get("recognition", {})

    # 配置文件 fallback: 仅在命令行未指定时使用
    if not args.rtsp and not args.mipi:
        source = video_cfg.get("source", "mipi")
        if source == "rtsp":
            args.rtsp = True
    if args.rtsp_url == "rtsp://192.168.144.25:8554/main.264":
        args.rtsp_url = video_cfg.get("rtsp_url", args.rtsp_url)
    if args.sensor_id == 0:
        args.sensor_id = int(mipi_cfg.get("sensor_id", args.sensor_id))
    if args.width == 1280:
        args.width = int(mipi_cfg.get("width", args.width))
    if args.height == 720:
        args.height = int(mipi_cfg.get("height", args.height))
    if args.fps_cap == 30:
        args.fps_cap = int(mipi_cfg.get("fps", args.fps_cap))
    if not args.fast:
        args.fast = recog_cfg.get("fast_mode", False)
        if isinstance(args.fast, str):
            args.fast = args.fast.lower() in ("true", "1", "yes")
    targets_dir = recog_cfg.get("targets_dir", "targets")

    # ROI 预设 (从配置文件)
    roi_cfg = recog_cfg.get("roi", None)
    initial_roi = None
    if roi_cfg and isinstance(roi_cfg, list) and len(roi_cfg) == 4:
        initial_roi = list(roi_cfg)
        log.info("配置文件预设 ROI: %s", initial_roi)

    # 追踪配置
    track_cfg = cfg.get("tracking", {})
    if args.auto_track or track_cfg.get("auto_track", False):
        args.auto_track = True
    dead_zone = float(track_cfg.get("dead_zone", 0.1))
    speed_gain = float(track_cfg.get("speed_gain", 50))
    max_lost_frames = int(track_cfg.get("max_lost_frames", 10))

    t_start = time.time()
    rec = TargetRecognizer(targets_dir=targets_dir)
    t_load = time.time()
    log.info("[阶段1] 模板加载+特征预计算: %.0f ms", (t_load - t_start)*1000)

    if args.batch:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        images = sorted(glob.glob("captures/*.jpg"))
        if not images:
            log.warning("captures/ 中没有图片")
            return
        log.info("批量测试 %d 张图片", len(images))
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            name = os.path.basename(img_path)
            results, timing = rec.recognize(img)
            log.info("--- %s ---", name)
            if results:
                for score, x, y, w, h, method in results:
                    log.info("  [%8s] 得分=%.3f 位置=(%d,%d) 大小=%dx%d", method, score, x, y, w, h)
            else:
                log.info("  未检测到目标")
            print_timing(timing)
            display = draw_results(img, results, timing, name)
            save_path = os.path.join(results_dir, f"result_{name}")
            cv2.imwrite(save_path, display)
            log.info("  保存: %s", save_path)
        log.info("所有结果已保存到 %s/", results_dir)

    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            log.warning("无法读取: %s", args.image)
            return
        results, timing = rec.recognize(img)
        log.info("找到 %d 个候选:", len(results))
        for score, x, y, w, h, method in results:
            log.info("  [%8s] 得分=%.3f 位置=(%d,%d) 大小=%dx%d", method, score, x, y, w, h)
        print_timing(timing)
        display = draw_results(img, results, timing)
        cv2.imshow("Result", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        import threading
        from datetime import datetime

        # ---------- 视频源初始化 ----------
        use_rtsp = args.rtsp
        use_mipi = False
        t_conn0 = time.time()

        if use_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            t_conn1 = time.time()
            log.info("[阶段2] RTSP 连接建立: %.0f ms", (t_conn1 - t_conn0)*1000)
            source_label = "RTSP"
            if not cap.isOpened():
                log.warning("无法连接 RTSP")
                return
        else:
            from mipi_camera import MIPICamera
            mipi_ok = False
            try:
                cap = MIPICamera(sensor_id=args.sensor_id,
                                 width=args.width, height=args.height,
                                 fps=args.fps_cap)
                if cap.isOpened():
                    mipi_ok = True
                else:
                    cap.release()
            except Exception as e:
                log.warning("MIPI 打开失败: %s", e)

            if mipi_ok:
                use_mipi = True
                t_conn1 = time.time()
                log.info("[阶段2] MIPI CSI 连接建立: %.0f ms", (t_conn1 - t_conn0)*1000)
                source_label = "MIPI"
            else:
                # 自动降级到 RTSP
                log.warning("MIPI 不可用, 自动降级到 RTSP: %s", args.rtsp_url)
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                t_conn1 = time.time()
                log.info("[阶段2] RTSP 连接建立 (降级): %.0f ms", (t_conn1 - t_conn0)*1000)
                source_label = "RTSP(fallback)"
                if not cap.isOpened():
                    log.warning("无法连接 RTSP, 退出")
                    return

        # 云台 SDK 初始化 (auto-track)
        cam = None
        if args.auto_track:
            try:
                from siyi_sdk import SIYIA8mini
                gimbal_cfg = cfg.get("gimbal", {})
                cam = SIYIA8mini(
                    ip=gimbal_cfg.get("ip", "192.168.144.25"),
                    port=int(gimbal_cfg.get("port", 37260)))
                log.info("云台 SDK 已连接 (auto-track)")
            except Exception as e:
                log.warning("云台 SDK 连接失败, 禁用 auto-track: %s", e)
                args.auto_track = False

        # 录像初始化
        writer = None
        if args.save:
            os.makedirs("recordings", exist_ok=True)
            fname = datetime.now().strftime("rec_%Y%m%d_%H%M%S.mp4")
            save_path = os.path.join("recordings", fname)
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, 25, (fw, fh))
            log.info("录像保存到: %s", save_path)

        # 共享状态
        lock = threading.Lock()
        latest_frame = [None]
        latest_frame_time = [0]
        latest_results = [[], {}]
        running = [True]
        use_fast = args.fast
        first_detect = [True]
        t_first_frame = [None]
        STALE_FRAME_THRESH = 0.5  # 超过500ms的帧视为过时

        # 追踪器状态
        tracker = [None]  # KalmanTracker instance
        recognize_interval = [1]  # 每N帧做一次全识别
        track_frame_count = [0]
        track_status = [""]  # OSD 追踪状态文本
        track_bbox = [None]  # 追踪框 (x, y, w, h)
        track_predicted = [False]  # 是否为预测框

        # ROI 状态
        roi = initial_roi  # [x, y, w, h] or None
        roi_drawing = False
        roi_start = None
        roi_end = None
        roi_temp = [None]

        # 系统状态缓存 (每秒更新)
        sys_stats_cache = [{}]
        sys_stats_time = [0]

        def mouse_callback(event, x, y, flags, param):
            nonlocal roi, roi_drawing, roi_start, roi_end
            if not roi_drawing:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_start = (x, y)
                roi_end = None
                roi_temp[0] = None
            elif event == cv2.EVENT_MOUSEMOVE and roi_start is not None:
                roi_end = (x, y)
                x1, y1 = roi_start
                x2, y2 = roi_end
                rx, ry = min(x1, x2), min(y1, y2)
                rw, rh = abs(x2 - x1), abs(y2 - y1)
                if rw > 10 and rh > 10:
                    roi_temp[0] = [rx, ry, rw, rh]
            elif event == cv2.EVENT_LBUTTONUP and roi_start is not None:
                roi_end = (x, y)
                x1, y1 = roi_start
                x2, y2 = roi_end
                rx, ry = min(x1, x2), min(y1, y2)
                rw, rh = abs(x2 - x1), abs(y2 - y1)
                if rw > 10 and rh > 10:
                    roi = [rx, ry, rw, rh]
                    log.info("ROI 设置: %s", roi)
                else:
                    log.info("ROI 太小, 忽略")
                roi_drawing = False
                roi_start = None
                roi_end = None
                roi_temp[0] = None

        def recognize_loop():
            """后台线程: 持续对最新帧做识别 + 追踪"""
            while running[0]:
                with lock:
                    frame = latest_frame[0]
                    frame_time = latest_frame_time[0]
                if frame is None:
                    time.sleep(0.01)
                    continue
                # 跳过过时的帧（>500ms前采集的）
                if frame_time > 0 and time.time() - frame_time > STALE_FRAME_THRESH:
                    time.sleep(0.005)
                    continue

                track_frame_count[0] += 1

                if tracker[0] and tracker[0].is_tracking():
                    # === 追踪模式 ===
                    detection = None
                    # 每 recognize_interval 帧做一次识别来校正追踪
                    if track_frame_count[0] % recognize_interval[0] == 0:
                        # ROI 裁剪
                        current_roi = roi
                        if current_roi:
                            rx, ry, rw, rh = current_roi
                            fh, fw = frame.shape[:2]
                            rx = max(0, min(rx, fw - 1))
                            ry = max(0, min(ry, fh - 1))
                            rw = min(rw, fw - rx)
                            rh = min(rh, fh - ry)
                            if rw > 10 and rh > 10:
                                scene = frame[ry:ry+rh, rx:rx+rw]
                                results, timing = rec.recognize(scene, fast=use_fast)
                                results = [(s, x+rx, y+ry, w, h, m) for s, x, y, w, h, m in results]
                            else:
                                results, timing = rec.recognize(frame, fast=use_fast)
                        else:
                            results, timing = rec.recognize(frame, fast=use_fast)

                        if results:
                            best = results[0]
                            detection = (best[1], best[2], best[3], best[4])
                        with lock:
                            latest_results[0] = results
                            latest_results[1] = timing

                    success, bbox = tracker[0].update(frame, detection)
                    if not success and not tracker[0].is_tracking():
                        # 追踪丢失，回退全帧识别
                        tracker[0] = None
                        recognize_interval[0] = 1
                        track_status[0] = ""
                        track_bbox[0] = None
                        track_predicted[0] = False
                        log.info("追踪器已释放, 回退全帧识别")
                    else:
                        track_bbox[0] = bbox
                        if success:
                            track_status[0] = "TRACKING"
                            track_predicted[0] = False
                            # 追踪稳定，逐步降低识别频率 (最多每5帧一次)
                            if tracker[0].lost_count == 0:
                                recognize_interval[0] = min(recognize_interval[0] + 1, 5)
                        else:
                            track_status[0] = "PREDICT"
                            track_predicted[0] = True

                    # 云台自动跟踪
                    if args.auto_track and cam and tracker[0] and tracker[0].is_tracking():
                        tx, ty, tw, th = tracker[0].bbox
                        tcx, tcy = tx + tw / 2, ty + th / 2
                        frame_cx = frame.shape[1] / 2
                        frame_cy = frame.shape[0] / 2
                        dx = (tcx - frame_cx) / frame_cx
                        dy = (tcy - frame_cy) / frame_cy
                        if abs(dx) > dead_zone or abs(dy) > dead_zone:
                            yaw_speed = int(np.clip(dx * speed_gain, -100, 100))
                            pitch_speed = int(np.clip(-dy * speed_gain, -100, 100))
                            cam.gimbal_rotate(yaw_speed, pitch_speed)
                        else:
                            cam.gimbal_rotate(0, 0)

                else:
                    # === 识别模式 ===
                    current_roi = roi
                    if current_roi:
                        rx, ry, rw, rh = current_roi
                        fh, fw = frame.shape[:2]
                        rx = max(0, min(rx, fw - 1))
                        ry = max(0, min(ry, fh - 1))
                        rw = min(rw, fw - rx)
                        rh = min(rh, fh - ry)
                        if rw > 10 and rh > 10:
                            scene = frame[ry:ry+rh, rx:rx+rw]
                            results, timing = rec.recognize(scene, fast=use_fast)
                            results = [(s, x+rx, y+ry, w, h, m) for s, x, y, w, h, m in results]
                        else:
                            results, timing = rec.recognize(frame, fast=use_fast)
                    else:
                        results, timing = rec.recognize(frame, fast=use_fast)

                    with lock:
                        latest_results[0] = results
                        latest_results[1] = timing

                    if results:
                        best = results[0]
                        bbox = (best[1], best[2], best[3], best[4])
                        tracker[0] = KalmanTracker(bbox, frame)
                        tracker[0].max_lost = max_lost_frames
                        recognize_interval[0] = 1
                        track_status[0] = "TRACKING"
                        track_bbox[0] = bbox
                        track_predicted[0] = False
                        log.info("目标锁定, 启动追踪器")
                    else:
                        track_status[0] = ""
                        track_bbox[0] = None
                        track_predicted[0] = False

                if first_detect[0] and latest_results[0]:
                    t_found = time.time()
                    log.info("[阶段4] 首次识别到目标: %.0f ms (从启动算起)", (t_found - t_start)*1000)
                    log.info("         识别耗时: %.0f ms", latest_results[1].get("total", 0)*1000)
                    log.info("         找到 %d 个候选", len(latest_results[0]))
                    first_detect[0] = False

        t = threading.Thread(target=recognize_loop, daemon=True)
        t.start()
        mode_str = "FAST" if use_fast else "FULL"
        cv2.namedWindow("A8mini Live")
        cv2.setMouseCallback("A8mini Live", mouse_callback)

        track_info = " [AUTO-TRACK]" if args.auto_track else ""
        log.info("实时识别中 [%s] [%s]%s... p=暂停  f=快速/全量  d=ROI  t=释放追踪  q=退出", source_label, mode_str, track_info)
        paused = False
        first_frame = True
        fps_counter = FPSCounter(window_size=30)

        reconnect_count = 0
        max_reconnect = 5
        consecutive_failures = 0

        while True:
            t_frame_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    log.warning('视频流断开，尝试重连 (%d/%d)', reconnect_count + 1, max_reconnect)
                    cap.release()
                    time.sleep(1)
                    try:
                        if use_mipi:
                            from mipi_camera import MIPICamera
                            cap = MIPICamera(sensor_id=args.sensor_id,
                                             width=args.width, height=args.height,
                                             fps=args.fps_cap)
                        else:
                            cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        reconnect_count += 1
                        consecutive_failures = 0
                        if reconnect_count >= max_reconnect:
                            log.error('达到最大重连次数 (%d), 退出', max_reconnect)
                            break
                        log.info('重连成功 (%d/%d)', reconnect_count, max_reconnect)
                    except Exception as e:
                        log.error('重连失败: %s', e)
                continue
            consecutive_failures = 0
            capture_latency_ms = (time.time() - t_frame_start) * 1000

            if first_frame:
                t_first_frame[0] = time.time()
                log.info("[阶段3] 收到第一帧: %.0f ms (从启动算起)", (t_first_frame[0] - t_start)*1000)
                first_frame = False

            # 送最新帧给识别线程
            if not paused:
                with lock:
                    latest_frame[0] = frame.copy()
                    latest_frame_time[0] = time.time()

            # 取最新识别结果叠加显示
            with lock:
                results = latest_results[0]
                timing = latest_results[1]

            if use_mipi:
                current_fps = cap.get_actual_fps()
            else:
                current_fps = fps_counter.tick()
            # 系统状态 (每秒更新)
            now_stats = time.time()
            if now_stats - sys_stats_time[0] > 1.0:
                sys_stats_cache[0] = get_system_stats()
                sys_stats_time[0] = now_stats

            status = f"{source_label} " + ("FAST" if use_fast else "FULL") + (" PAUSED" if paused else "")
            if roi_drawing:
                status += " [ROI绘制中...]"
            elif roi:
                status += " [ROI]"
            display_roi = roi_temp[0] if roi_drawing and roi_temp[0] else roi
            display = draw_results(frame, results, timing, status,
                                   fps=current_fps, latency_ms=capture_latency_ms,
                                   roi=display_roi, system_stats=sys_stats_cache[0])

            # 追踪框 OSD
            t_bbox = track_bbox[0]
            t_status = track_status[0]
            if t_bbox is not None and t_status:
                tx, ty, tw, th = t_bbox
                if t_status == "TRACKING":
                    t_color = (0, 255, 0)  # 绿色
                else:  # PREDICT
                    t_color = (0, 255, 255)  # 黄色
                cv2.rectangle(display, (tx, ty), (tx + tw, ty + th), t_color, 3)
                cv2.putText(display, t_status, (tx, ty - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, t_color, 2)
                # 显示识别间隔
                intv_text = f"recog every {recognize_interval[0]}f"
                cv2.putText(display, intv_text, (tx, ty + th + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, t_color, 1)

            # 写入录像（带识别框的画面）
            if writer is not None:
                writer.write(display)

            cv2.imshow("A8mini Live", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                log.info("暂停" if paused else "继续")
            elif key == ord('f'):
                use_fast = not use_fast
                log.info("切换到 %s 模式", "快速(ORB+颜色)" if use_fast else "全量(5方法)")
            elif key == ord('d'):
                if roi_drawing:
                    roi_drawing = False
                    roi_start = None
                    roi_end = None
                    roi_temp[0] = None
                    log.info("ROI 绘制取消")
                elif roi:
                    roi = None
                    log.info("ROI 已清除")
                else:
                    roi_drawing = True
                    log.info("ROI 绘制模式: 鼠标拖框, d/ESC取消")
            elif key == ord('t'):
                if tracker[0]:
                    tracker[0] = None
                    track_status[0] = ""
                    track_bbox[0] = None
                    track_predicted[0] = False
                    recognize_interval[0] = 1
                    if cam:
                        cam.gimbal_rotate(0, 0)
                    log.info("追踪器已手动释放")
            elif key == 27:  # ESC
                if roi_drawing:
                    roi_drawing = False
                    roi_start = None
                    roi_end = None
                    roi_temp[0] = None
                    log.info("ROI 绘制取消")

        running[0] = False
        t.join(timeout=2)
        if writer is not None:
            writer.release()
            log.info("录像已保存")
        if cam:
            cam.gimbal_rotate(0, 0)  # 停止云台
            cam.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
