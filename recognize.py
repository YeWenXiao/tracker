"""
目标识别引擎 (多方法 + 分步计时)
方法:
  1. 多尺度模板匹配
  2. ORB 特征点匹配 + 单应性
  3. SIFT 特征点匹配 + 单应性
  4. HSV 颜色直方图反投影
  5. 边缘轮廓匹配

用法:
  python recognize.py --batch              # 批量测试
  python recognize.py --image xxx.jpg      # 单张测试
  python recognize.py                      # RTSP实时 (全方法, ~1fps)
  python recognize.py --fast               # RTSP实时 快速模式 (ORB+颜色, ~30fps)
  python recognize.py --save               # RTSP实时 + 录像保存
  python recognize.py --fast --save        # 快速模式 + 录像保存
  python recognize.py --fast --api         # 快速模式 + HTTP API 服务器
"""

import cv2
import numpy as np
import os
import json
import argparse
import time
import glob
import queue


# 全局 SSE 事件队列列表（每个 SSE 客户端一个队列）
_sse_clients = []

def push_event(event):
    """向所有 SSE 客户端推送事件"""
    dead = []
    for q in _sse_clients:
        try:
            q.put_nowait(event)
        except queue.Full:
            dead.append(q)
    for q in dead:
        _sse_clients.remove(q)

def register_sse_client():
    """注册新的 SSE 客户端，返回事件队列"""
    q = queue.Queue(maxsize=100)
    _sse_clients.append(q)
    return q

def unregister_sse_client(q):
    """注销 SSE 客户端"""
    if q in _sse_clients:
        _sse_clients.remove(q)


class TargetRecognizer:
    def __init__(self, targets_dir="targets"):
        self.targets = []
        self._targets_dir = targets_dir
        self._last_reload_time = None
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)
        # SIFT 用 FLANN 匹配器
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self._load(targets_dir)


    def _prepare_single_target(self, targets_dir, ann):
        """为单个模板计算所有特征，返回特征字典；失败返回 None"""
        img = cv2.imread(os.path.join(targets_dir, ann["crop"]))
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ORB 特征
        orb_kp, orb_des = self.orb.detectAndCompute(gray, None)
        # SIFT 特征
        sift_kp, sift_des = self.sift.detectAndCompute(gray, None)
        # HSV 直方图（用于颜色验证）
        hist_hs = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist_hs, hist_hs)
        # HSV 直方图（用于反投影）
        hist_bp = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_bp, hist_bp, 0, 255, cv2.NORM_MINMAX)
        # 边缘
        edges = cv2.Canny(gray, 50, 150)

        # 读取目标权重和最低置信度（默认 weight=1.0, min_confidence=0.45）
        weight = float(ann.get("weight", 1.0))
        min_confidence = float(ann.get("min_confidence", 0.45))

        return {
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
            "weight": weight,
            "min_confidence": min_confidence,
        }

    def _prepare_targets(self, targets_dir):
        """准备目标模板列表（不修改 self.targets），返回新列表"""
        info_path = os.path.join(targets_dir, "target_info.json")
        with open(info_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        new_targets = []
        for ann in annotations:
            target = self._prepare_single_target(targets_dir, ann)
            if target:
                new_targets.append(target)
        return new_targets

    def _load(self, targets_dir):
        self._targets_dir = targets_dir
        self.targets = self._prepare_targets(targets_dir)
        self._last_reload_time = time.time()
        print(f"已加载 {len(self.targets)} 个目标模板")

    def reload_targets(self, targets_dir=None):
        """增量热加载：只计算新增模板的特征，保留未变化的模板"""
        if targets_dir is None:
            targets_dir = self._targets_dir
        try:
            info_path = os.path.join(targets_dir, "target_info.json")
            with open(info_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)

            # 检查哪些模板是新的、哪些被删除
            old_names = {t["name"] for t in self.targets}
            new_names = {ann["crop"] for ann in annotations}

            added = new_names - old_names
            removed = old_names - new_names
            kept = old_names & new_names

            if not added and not removed:
                print("[热加载] 无变化")
                return

            # 保留未变化的模板（用字典加速查找）
            kept_map = {t["name"]: t for t in self.targets if t["name"] in kept}

            # 只计算新增模板的特征
            added_targets = {}
            for ann in annotations:
                if ann["crop"] in added:
                    target = self._prepare_single_target(targets_dir, ann)
                    if target:
                        added_targets[ann["crop"]] = target

            # 按 annotations 原始顺序重组
            all_targets = []
            for ann in annotations:
                name = ann["crop"]
                if name in kept_map:
                    all_targets.append(kept_map[name])
                elif name in added_targets:
                    all_targets.append(added_targets[name])

            # 原子替换（Python GIL 保证赋值原子性）
            self.targets = all_targets
            self._last_reload_time = time.time()
            print(f"[热加载] +{len(added)} -{len(removed)} 保留{len(kept)} = 共{len(all_targets)} 个模板")
            # 推送 SSE 重载事件
            push_event({
                "type": "reload",
                "added": len(added),
                "removed": len(removed),
                "count": len(all_targets),
                "time": self._last_reload_time,
            })
        except Exception as e:
            print(f"[热加载] 失败: {e}")
    def recognize(self, scene_bgr, fast=False):
        """
        在场景中识别目标
        fast=True: 降分辨率模板 + ORB + 颜色反投影 (~160ms, ~6fps)
        fast=False: 全部5种方法 (~1000ms, ~1fps)
        返回: results_list, timing_dict
        """
        scene_h, scene_w = scene_bgr.shape[:2]
        scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
        scene_hsv = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2HSV)
        timing = {}
        all_results = []

        # 取本地引用，避免识别过程中被替换导致不一致
        targets = self.targets
        if not targets:
            timing["total"] = 0
            return [], timing

        aspect_ratios = [t["gray"].shape[0] / t["gray"].shape[1] for t in targets]
        avg_ar = np.mean(aspect_ratios)
        min_dim = int(min(scene_w, scene_h) * 0.03)

        # === 方法1: 多尺度模板匹配 ===
        t0 = time.time()
        template_results = []
        if fast:
            # 快速模式: 场景缩小到1/3，减少尺度数
            ds = 1.0 / 3
            scene_small = cv2.resize(scene_gray, None, fx=ds, fy=ds)
            scales = np.arange(0.3, 1.5, 0.4)  # 3个尺度 vs 全量12个
        else:
            scene_small = scene_gray
            ds = 1.0
            scales = np.arange(0.3, 1.5, 0.1)
        for t in targets:
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
                for t in targets:
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
                for t in targets:
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
            for t in targets:
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
            for t in targets:
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
            # 对每个目标计算颜色相似度，找到最佳匹配的目标
            best_color_score = -1
            best_target = targets[0] if targets else None
            for t in targets:
                cs = cv2.compareHist(t["hist_hs"], hist, cv2.HISTCMP_CORREL)
                if cs > best_color_score:
                    best_color_score = cs
                    best_target = t
            # 使用最佳匹配目标的 min_confidence 和 weight
            min_conf = best_target.get("min_confidence", 0.45) if best_target else 0.45
            weight = best_target.get("weight", 1.0) if best_target else 1.0
            combined = 0.4 * score + 0.6 * max(best_color_score, 0)
            if combined >= min_conf:
                # 最终得分乘以目标权重
                weighted_score = combined * weight
                verified.append((weighted_score, x, y, x2-x, y2-y, method))

        verified.sort(key=lambda r: r[0], reverse=True)
        timing["nms_verify"] = time.time() - t0
        timing["total"] = sum(timing.values())

        return verified[:5], timing

    def _homography_box(self, t, good_matches, scene_kps, kp_key,
                        scene_w, scene_h, min_dim, avg_ar):
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


def draw_results(img, results, timing=None, label="", recognizer=None):
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
    if timing:
        for key in ["template", "orb", "sift", "color_bp", "edge", "nms_verify", "total"]:
            if key in timing:
                prefix = ">> " if key == "total" else "   "
                text = f"{prefix}{key}: {timing[key]*1000:.0f}ms"
                cv2.putText(display, text, (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_off += 20

    # 右上角: 目标数量和最后更新时间
    if recognizer is not None:
        disp_h, disp_w = display.shape[:2]
        n_targets = len(recognizer.targets)
        target_text = f"目标: {n_targets}"
        cv2.putText(display, target_text, (disp_w - 200, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if recognizer._last_reload_time is not None:
            from datetime import datetime
            t_str = datetime.fromtimestamp(recognizer._last_reload_time).strftime("%H:%M:%S")
            reload_text = f"更新: {t_str}"
            cv2.putText(display, reload_text, (disp_w - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return display


def print_timing(timing):
    print("  分步耗时:")
    for key in ["template", "orb", "sift", "color_bp", "edge", "nms_verify"]:
        if key in timing:
            print(f"    {key:12s}: {timing[key]*1000:6.0f} ms")
    total_ms = timing.get('total', 0) * 1000
    total_label = "总计"
    print(f"    {total_label:12s}: {total_ms:6.0f} ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="测试图片路径")
    parser.add_argument("--batch", action="store_true", help="批量测试captures/下所有图片")
    parser.add_argument("--fast", action="store_true", help="快速模式: 只用ORB+颜色 (~20ms)")
    parser.add_argument("--save", action="store_true", help="保存识别视频到 recordings/")
    parser.add_argument("--rtsp", default="rtsp://192.168.144.25:8554/main.264")
    parser.add_argument("--api", action="store_true", help="启动 HTTP API 服务器 (端口5000)")
    parser.add_argument("--api-port", type=int, default=5000, help="API端口")
    args = parser.parse_args()

    t_start = time.time()
    rec = TargetRecognizer()
    t_load = time.time()
    print(f"[阶段1] 模板加载+特征预计算: {(t_load - t_start)*1000:.0f} ms")

    if args.batch:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        张图片 = sorted(glob.glob("captures/*.jpg"))
        if not 张图片:
            print("captures/ 中没有图片")
            return
        print(f"批量测试 {len(images)} 张图片\n")
        for img_path in 张图片:
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

        # 启动 HTTP API 服务器（可选）
        if args.api:
            from target_server import run_server, set_recognizer
            set_recognizer(rec)
            api_thread = threading.Thread(target=run_server,
                                          kwargs={"port": args.api_port},
                                          daemon=True)
            api_thread.start()
            print(f"[API] HTTP 服务器已启动: http://0.0.0.0:{args.api_port}")

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        t_conn0 = time.time()
        cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        t_conn1 = time.time()
        print(f"[阶段2] RTSP连接建立: {(t_conn1 - t_conn0)*1000:.0f} ms")
        if not cap.isOpened():
            print("无法连接")
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
                # 推送识别结果 SSE 事件
                if results:
                    best = results[0]
                    push_event({
                        "type": "detection",
                        "score": round(best[0], 4),
                        "x": best[1], "y": best[2],
                        "w": best[3], "h": best[4],
                        "method": best[5],
                        "count": len(results),
                        "time": time.time(),
                    })
                if first_detect[0] and results:
                    t_found = time.time()
                    print(f"[阶段4] 首次识别到目标: {(t_found - t_start)*1000:.0f} ms (从启动算起)")
                    print(f"         识别耗时: {timing.get('total',0)*1000:.0f} ms")
                    print(f"         找到 {len(results)} 个候选")
                    first_detect[0] = False

        def target_watcher():
            """每2秒检查 targets/target_info.json 的修改时间"""
            info_path = os.path.join(rec._targets_dir, "target_info.json")
            try:
                last_mtime = os.path.getmtime(info_path)
            except OSError:
                last_mtime = 0
            while running[0]:
                time.sleep(2)
                try:
                    mtime = os.path.getmtime(info_path)
                    if mtime != last_mtime:
                        last_mtime = mtime
                        rec.reload_targets()
                except Exception:
                    pass

        t = threading.Thread(target=recognize_loop, daemon=True)
        t.start()

        tw = threading.Thread(target=target_watcher, daemon=True)
        tw.start()

        mode_str = "FAST" if use_fast else "FULL"
        print(f"实时识别中 [{mode_str}]... p=暂停/继续  f=切换快速/全量  r=重载目标  q=退出")
        paused = False
        first_frame = True

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

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

            status = ("FAST" if use_fast else "FULL") + (" PAUSED" if paused else "")
            display = draw_results(frame, results, timing, status, recognizer=rec)

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
                mode = "快速(ORB+颜色)" if use_fast else "全量(5方法)"
                print(f"切换到 {mode} 模式")
            elif key == ord('r'):
                print("手动触发目标重载...")
                threading.Thread(target=rec.reload_targets, daemon=True).start()

        running[0] = False
        t.join(timeout=2)
        if writer is not None:
            writer.release()
            print("录像已保存")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
