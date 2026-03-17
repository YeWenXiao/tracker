"""
Target Recognition Engine (Multi-method + Timing)
Methods:
  1. Multi-scale template matching
  2. ORB feature matching + homography
  3. SIFT feature matching + homography
  4. HSV color histogram backprojection
  5. Edge contour matching

Usage:
  python recognize.py --batch              # batch test
  python recognize.py --image xxx.jpg      # single image
  python recognize.py                      # RTSP live (all methods, ~1fps)
  python recognize.py --fast               # RTSP live fast (ORB+color, ~30fps)
  python recognize.py --save               # RTSP live + record
  python recognize.py --fast --save        # fast + record
"""

import cv2
import numpy as np
import os
import json
import argparse
import time
import glob


class TargetRecognizer:
    def __init__(self, targets_dir="targets"):
        self.targets = []
        self._targets_dir = targets_dir
        self._last_reload_time = None
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)
        # FLANN for SIFT
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self._load(targets_dir)


    def _prepare_single_target(self, targets_dir, ann):
        """Compute all features for a single template, return dict or None on failure"""
        img = cv2.imread(os.path.join(targets_dir, ann["crop"]))
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ORB
        orb_kp, orb_des = self.orb.detectAndCompute(gray, None)
        # SIFT
        sift_kp, sift_des = self.sift.detectAndCompute(gray, None)
        # HSV histogram
        hist_hs = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist_hs, hist_hs)
        # Backprojection histogram
        hist_bp = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_bp, hist_bp, 0, 255, cv2.NORM_MINMAX)
        # Edges
        edges = cv2.Canny(gray, 50, 150)

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
        }

    def _prepare_targets(self, targets_dir):
        """Prepare target template list (does not modify self.targets), returns new list"""
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
        print(f"Loaded {len(self.targets)} target templates")

    def reload_targets(self, targets_dir=None):
        """Incremental hot-reload: only compute features for new templates"""
        if targets_dir is None:
            targets_dir = self._targets_dir
        try:
            info_path = os.path.join(targets_dir, "target_info.json")
            with open(info_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)

            # Check which templates are new / removed
            old_names = {t["name"] for t in self.targets}
            new_names = {ann["crop"] for ann in annotations}

            added = new_names - old_names
            removed = old_names - new_names
            kept = old_names & new_names

            if not added and not removed:
                print("[Hot-reload] No changes")
                return

            # Keep unchanged templates (dict for fast lookup)
            kept_map = {t["name"]: t for t in self.targets if t["name"] in kept}

            # Only compute features for new templates
            added_targets = {}
            for ann in annotations:
                if ann["crop"] in added:
                    target = self._prepare_single_target(targets_dir, ann)
                    if target:
                        added_targets[ann["crop"]] = target

            # Reassemble in annotation order
            all_targets = []
            for ann in annotations:
                name = ann["crop"]
                if name in kept_map:
                    all_targets.append(kept_map[name])
                elif name in added_targets:
                    all_targets.append(added_targets[name])

            # Atomic swap (Python GIL guarantees assignment atomicity)
            self.targets = all_targets
            self._last_reload_time = time.time()
            print(f"[Hot-reload] +{len(added)} -{len(removed)} kept={len(kept)} total={len(all_targets)}")
        except Exception as e:
            print(f"[Hot-reload] Failed: {e}")
    def recognize(self, scene_bgr, fast=False):
        """
        Recognize targets in scene
        fast=True: downscaled template + ORB + color backprojection
        fast=False: all 5 methods
        Returns: results_list, timing_dict
        """
        scene_h, scene_w = scene_bgr.shape[:2]
        scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
        scene_hsv = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2HSV)
        timing = {}
        all_results = []

        # Local ref to avoid inconsistency during hot-swap
        targets = self.targets
        if not targets:
            timing["total"] = 0
            return [], timing

        aspect_ratios = [t["gray"].shape[0] / t["gray"].shape[1] for t in targets]
        avg_ar = np.mean(aspect_ratios)
        min_dim = int(min(scene_w, scene_h) * 0.03)

        # === Method 1: Multi-scale template matching ===
        t0 = time.time()
        template_results = []
        if fast:
            ds = 1.0 / 3
            scene_small = cv2.resize(scene_gray, None, fx=ds, fy=ds)
            scales = np.arange(0.3, 1.5, 0.4)
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
                    ox, oy = int(max_loc[0] / ds), int(max_loc[1] / ds)
                    ow, oh = int(nw / ds), int(nh / ds)
                    template_results.append((max_val, ox, oy, ow, oh, "template"))
        timing["template"] = time.time() - t0
        all_results.extend(template_results)

        high_conf = fast and any(s > 0.8 for s, *_ in template_results)

        # === Method 2: ORB feature matching ===
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
            # === Method 3: SIFT feature matching (FLANN) ===
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

        # === Method 4: HSV color backprojection ===
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
            # === Method 5: Edge template matching ===
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

        # === NMS + color verification ===
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
                for t in targets
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
    # Top-left info
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

    # Top-right: target count and last update time
    if recognizer is not None:
        disp_h, disp_w = display.shape[:2]
        n_targets = len(recognizer.targets)
        target_text = f"Targets: {n_targets}"
        cv2.putText(display, target_text, (disp_w - 200, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if recognizer._last_reload_time is not None:
            from datetime import datetime
            t_str = datetime.fromtimestamp(recognizer._last_reload_time).strftime("%H:%M:%S")
            reload_text = f"Updated: {t_str}"
            cv2.putText(display, reload_text, (disp_w - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return display


def print_timing(timing):
    print("  Timing breakdown:")
    for key in ["template", "orb", "sift", "color_bp", "edge", "nms_verify"]:
        if key in timing:
            print(f"    {key:12s}: {timing[key]*1000:6.0f} ms")
    total_ms = timing.get('total', 0) * 1000
    total_label = "total"
    print(f"    {total_label:12s}: {total_ms:6.0f} ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--rtsp", default="rtsp://192.168.144.25:8554/main.264")
    args = parser.parse_args()

    t_start = time.time()
    rec = TargetRecognizer()
    t_load = time.time()
    print(f"[Phase 1] Template load + feature precompute: {(t_load - t_start)*1000:.0f} ms")

    if args.batch:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        images = sorted(glob.glob("captures/*.jpg"))
        if not images:
            print("No images in captures/")
            return
        print(f"Batch testing {len(images)} images\n")
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            name = os.path.basename(img_path)
            results, timing = rec.recognize(img)
            print(f"--- {name} ---")
            if results:
                for score, x, y, w, h, method in results:
                    print(f"  [{method:8s}] score={score:.3f} pos=({x},{y}) size={w}x{h}")
            else:
                print("  No target detected")
            print_timing(timing)
            display = draw_results(img, results, timing, name)
            save_path = os.path.join(results_dir, f"result_{name}")
            cv2.imwrite(save_path, display)
            print(f"  Saved: {save_path}\n")
        print(f"All results saved to {results_dir}/")

    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Cannot read: {args.image}")
            return
        results, timing = rec.recognize(img)
        print(f"Found {len(results)} candidates:")
        for score, x, y, w, h, method in results:
            print(f"  [{method:8s}] score={score:.3f} pos=({x},{y}) size={w}x{h}")
        print_timing(timing)
        display = draw_results(img, results, timing)
        cv2.imshow("Result", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        import threading
        from datetime import datetime

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        t_conn0 = time.time()
        cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        t_conn1 = time.time()
        print(f"[Phase 2] RTSP connected: {(t_conn1 - t_conn0)*1000:.0f} ms")
        if not cap.isOpened():
            print("Cannot connect")
            return

        writer = None
        if args.save:
            os.makedirs("recordings", exist_ok=True)
            fname = datetime.now().strftime("rec_%Y%m%d_%H%M%S.mp4")
            save_path = os.path.join("recordings", fname)
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, 25, (fw, fh))
            print(f"Recording to: {save_path}")

        lock = threading.Lock()
        latest_frame = [None]
        latest_results = [[], {}]
        running = [True]
        use_fast = args.fast
        first_detect = [True]
        t_first_frame = [None]

        def recognize_loop():
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
                    print(f"[Phase 4] First detection: {(t_found - t_start)*1000:.0f} ms")
                    print(f"         Recognition time: {timing.get('total',0)*1000:.0f} ms")
                    print(f"         Found {len(results)} candidates")
                    first_detect[0] = False

        def target_watcher():
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
        print(f"Live recognition [{mode_str}]... p=pause  f=toggle fast/full  r=reload targets  q=quit")
        paused = False
        first_frame = True

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if first_frame:
                t_first_frame[0] = time.time()
                print(f"[Phase 3] First frame: {(t_first_frame[0] - t_start)*1000:.0f} ms")
                first_frame = False

            if not paused:
                with lock:
                    latest_frame[0] = frame.copy()

            with lock:
                results = latest_results[0]
                timing = latest_results[1]

            status = ("FAST" if use_fast else "FULL") + (" PAUSED" if paused else "")
            display = draw_results(frame, results, timing, status, recognizer=rec)

            if writer is not None:
                writer.write(display)

            cv2.imshow("A8mini Live", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('f'):
                use_fast = not use_fast
                mode = "Fast(ORB+color)" if use_fast else "Full(5 methods)"
                print(f"Switched to {mode}")
            elif key == ord('r'):
                print("Manual target reload triggered...")
                threading.Thread(target=rec.reload_targets, daemon=True).start()

        running[0] = False
        t.join(timeout=2)
        if writer is not None:
            writer.release()
            print("Recording saved")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
