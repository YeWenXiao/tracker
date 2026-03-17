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
"""

import cv2
import numpy as np
import os
import json
import argparse
import time
import glob


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
        if self.use_cuda:
            print('[CUDA] GPU 加速已启用 (cvtColor/resize)')
        else:
            print('[CUDA] GPU 不可用, 使用 CPU 模式')
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
        print(f"已加载 {len(self.targets)} 个目标模板")

    def recognize(self, scene_bgr, fast=False):
        """
        在场景中识别目标
        fast=True: 降分辨率模板 + ORB + 颜色反投影 (~160ms, ~6fps)
        fast=False: 全部5种方法 (~1000ms, ~1fps)
        返回: results_list, timing_dict
        """
        scene_h, scene_w = scene_bgr.shape[:2]
        if self.use_cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(scene_bgr)
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            scene_gray = gpu_gray.download()
            gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
            scene_hsv = gpu_hsv.download()
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
                res = cv2.matchTemplate(scene_small, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.6:
                    # 坐标映射回原始分辨率
                    ox, oy = int(max_loc[0] / ds), int(max_loc[1] / ds)
                    ow, oh = int(nw / ds), int(nh / ds)
                    template_results.append((max_val, ox, oy, ow, oh, "template"))
        timing["template"] = time.time() - t0
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


def draw_results(img, results, timing=None, label="", fps=None, latency_ms=None):
    """在图上画识别结果和计时信息"""
    display = img.copy()
    for score, x, y, w, h, method in results:
        color = (0, 255, 0) if score > 0.6 else (0, 255, 255) if score > 0.4 else (0, 0, 255)
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
        cv2.putText(display, f"{method} {score:.2f}", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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
    return display


def print_timing(timing):
    """打印分步耗时"""
    print(f"  分步耗时:")
    for key in ["template", "orb", "sift", "color_bp", "edge", "nms_verify"]:
        if key in timing:
            print(f"    {key:12s}: {timing[key]*1000:6.0f} ms")
    print(f"    {'总计':12s}: {timing.get('total',0)*1000:6.0f} ms")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="测试图片路径")
    parser.add_argument("--batch", action="store_true", help="批量测试captures/下所有图片")
    parser.add_argument("--fast", action="store_true", help="快速模式: 只用ORB+颜色 (~20ms)")
    parser.add_argument("--save", action="store_true", help="保存识别视频到 recordings/")
    parser.add_argument("--rtsp", action="store_true",
                        help="使用 RTSP 视频源 (默认使用 MIPI CSI)")
    parser.add_argument("--rtsp-url", default="rtsp://192.168.144.25:8554/main.264",
                        help="RTSP 地址 (仅 --rtsp 时使用)")
    parser.add_argument("--mipi", action="store_true",
                        help="使用 MIPI CSI 摄像头 (默认已启用)")
    parser.add_argument("--sensor-id", type=int, default=0,
                        help="MIPI 摄像头编号 (默认0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="MIPI 采集宽度 (默认1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="MIPI 采集高度 (默认720)")
    parser.add_argument("--fps-cap", type=int, default=30,
                        help="MIPI 采集帧率 (默认30)")
    args = parser.parse_args()

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

    t_start = time.time()
    rec = TargetRecognizer(targets_dir=targets_dir)
    t_load = time.time()
    print(f"[阶段1] 模板加载+特征预计算: {(t_load - t_start)*1000:.0f} ms")

    if args.batch:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        images = sorted(glob.glob("captures/*.jpg"))
        if not images:
            print("captures/ 中没有图片")
            return
        print(f"批量测试 {len(images)} 张图片\n")
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            name = os.path.basename(img_path)
            results, timing = rec.recognize(img)
            print(f"--- {name} ---")
            if results:
                for score, x, y, w, h, method in results:
                    print(f"  [{method:8s}] 得分={score:.3f} 位置=({x},{y}) 大小={w}x{h}")
            else:
                print("  未检测到目标")
            print_timing(timing)
            display = draw_results(img, results, timing, name)
            save_path = os.path.join(results_dir, f"result_{name}")
            cv2.imwrite(save_path, display)
            print(f"  保存: {save_path}\n")
        print(f"所有结果已保存到 {results_dir}/")

    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"无法读取: {args.image}")
            return
        results, timing = rec.recognize(img)
        print(f"找到 {len(results)} 个候选:")
        for score, x, y, w, h, method in results:
            print(f"  [{method:8s}] 得分={score:.3f} 位置=({x},{y}) 大小={w}x{h}")
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
        t_conn0 = time.time()

        if use_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            t_conn1 = time.time()
            print(f"[阶段2] RTSP 连接建立: {(t_conn1 - t_conn0)*1000:.0f} ms")
            source_label = "RTSP"
            if not cap.isOpened():
                print("无法连接 RTSP")
                return
        else:
            from mipi_camera import MIPICamera
            cap = MIPICamera(sensor_id=args.sensor_id,
                             width=args.width, height=args.height,
                             fps=args.fps_cap)
            t_conn1 = time.time()
            print(f"[阶段2] MIPI CSI 连接建立: {(t_conn1 - t_conn0)*1000:.0f} ms")
            source_label = "MIPI"
            if not cap.isOpened():
                print("无法打开 MIPI CSI 摄像头")
                return

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
            print(f"录像保存到: {save_path}")

        # 共享状态
        lock = threading.Lock()
        latest_frame = [None]
        latest_results = [[], {}]
        running = [True]
        use_fast = args.fast
        first_detect = [True]
        t_first_frame = [None]

        def recognize_loop():
            """后台线程: 持续对最新帧做识别"""
            while running[0]:
                with lock:
                    frame = latest_frame[0]
                if frame is None:
                    time.sleep(0.01)
                    continue
                results, timing = rec.recognize(frame, fast=use_fast)
                with lock:
                    latest_results[0] = results
                    latest_results[1] = timing
                if first_detect[0] and results:
                    t_found = time.time()
                    print(f"[阶段4] 首次识别到目标: {(t_found - t_start)*1000:.0f} ms (从启动算起)")
                    print(f"         识别耗时: {timing.get('total',0)*1000:.0f} ms")
                    print(f"         找到 {len(results)} 个候选")
                    first_detect[0] = False

        t = threading.Thread(target=recognize_loop, daemon=True)
        t.start()
        mode_str = "FAST" if use_fast else "FULL"
        print(f"实时识别中 [{source_label}] [{mode_str}]... p=暂停/继续  f=切换快速/全量  q=退出")
        paused = False
        first_frame = True
        fps_counter = FPSCounter(window_size=30)

        while True:
            t_frame_start = time.time()
            if mipi_cam is not None:
                frame = mipi_cam.read()
                if frame is None:
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    continue
            capture_latency_ms = (time.time() - t_frame_start) * 1000

            if first_frame:
                t_first_frame[0] = time.time()
                print(f"[阶段3] 收到第一帧: {(t_first_frame[0] - t_start)*1000:.0f} ms (从启动算起)")
                first_frame = False

            # 送最新帧给识别线程
            if not paused:
                with lock:
                    latest_frame[0] = frame.copy()

            # 取最新识别结果叠加显示
            with lock:
                results = latest_results[0]
                timing = latest_results[1]

            if mipi_cam is not None:
                current_fps = mipi_cam.get_fps()
            else:
                current_fps = fps_counter.tick()
            status = f"{source_label} " + ("FAST" if use_fast else "FULL") + (" PAUSED" if paused else "")
            display = draw_results(frame, results, timing, status,
                                   fps=current_fps, latency_ms=capture_latency_ms)

            # 写入录像（带识别框的画面）
            if writer is not None:
                writer.write(display)

            cv2.imshow("A8mini Live", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("暂停" if paused else "继续")
            elif key == ord('f'):
                use_fast = not use_fast
                print(f"切换到 {'快速(ORB+颜色)' if use_fast else '全量(5方法)'} 模式")

        running[0] = False
        t.join(timeout=2)
        if writer is not None:
            writer.release()
            print(f"录像已保存")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
