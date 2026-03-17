"""
实时视频流目标标注工具 (v1.5)

在 RTSP 实时视频流上直接框选新目标，保存到 targets/ 并触发热加载。
无需停止识别程序，无需断开视频流。

用法:
  python annotate_live.py                    # 连接RTSP流，框选目标
  python annotate_live.py --rtsp URL         # 指定RTSP地址
  python annotate_live.py --recognizer-url http://localhost:5000  # 通过API触发重载

操作:
  鼠标拖框 -> 选择目标区域
  s = 保存框选区域为新目标并触发重载
  c = 清除当前框选
  z = 暂停/继续视频（方便精确框选）
  q = 退出
"""

import cv2
import os
import json
import argparse
import time
import threading
from datetime import datetime

try:
    import requests as http_requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


TARGETS_DIR = "targets"


class LiveAnnotator:
    def __init__(self, rtsp_url, targets_dir=TARGETS_DIR, recognizer_url=None):
        self.rtsp_url = rtsp_url
        self.targets_dir = targets_dir
        self.recognizer_url = recognizer_url
        os.makedirs(targets_dir, exist_ok=True)

        # 框选状态
        self.drawing = False
        self.ix, self.iy = 0, 0
        self.rect = None
        self.current_frame = None
        self.paused = False
        self.frozen_frame = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.rect = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            pass  # 绘制在主循环中处理
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.rect = (x1, y1, x2, y2)

    def save_target(self, frame):
        """保存框选区域为新目标，更新 target_info.json，触发重载"""
        if self.rect is None:
            print("没有框选区域")
            return False

        x1, y1, x2, y2 = self.rect
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            print("裁剪区域为空")
            return False

        # 读取现有标注
        info_path = os.path.join(self.targets_dir, "target_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
        else:
            annotations = []

        # 保存裁剪图
        idx = len(annotations)
        crop_name = f"target_{idx:03d}.jpg"
        crop_path = os.path.join(self.targets_dir, crop_name)
        cv2.imwrite(crop_path, crop)

        h, w = frame.shape[:2]
        annotations.append({
            "source": f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "crop": crop_name,
            "bbox": [x1, y1, x2, y2],
            "image_size": [w, h]
        })

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        print(f"已保存: {crop_path} ({x2-x1}x{y2-y1})")

        # 触发识别引擎重载
        self._trigger_reload()

        self.rect = None
        return True

    def _trigger_reload(self):
        """通过 HTTP API 或本地方式触发识别引擎重载"""
        if self.recognizer_url:
            if not HAS_REQUESTS:
                print("[警告] requests 未安装，无法通过API触发重载")
                print("  安装: pip install requests")
                print("  或手动在识别窗口按 r 重载")
                return
            try:
                url = f"{self.recognizer_url.rstrip('/')}/targets/reload"
                resp = http_requests.post(url, timeout=5)
                data = resp.json()
                print(f"[API重载] {data}")
            except Exception as e:
                print(f"[API重载失败] {e}")
                print("  请在识别窗口按 r 手动重载")
        else:
            print("[提示] targets/ 已更新。如果识别程序使用了 --watch，会自动重载。")
            print("  否则请在识别窗口按 r 手动重载。")
            print("  或使用 --recognizer-url http://localhost:5000 通过API触发。")

    def run(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"无法连接 RTSP: {self.rtsp_url}")
            return

        cv2.namedWindow("Live Annotate")
        cv2.setMouseCallback("Live Annotate", self.mouse_callback)

        print("实时标注模式:")
        print("  鼠标拖框选择目标 -> s=保存 -> 自动触发识别引擎重载")
        print("  z=暂停/继续  c=清除框选  q=退出")

        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    continue
                self.current_frame = frame.copy()
            else:
                if self.frozen_frame is None:
                    self.frozen_frame = self.current_frame.copy() if self.current_frame is not None else None
                frame = self.frozen_frame

            if frame is None:
                continue

            display = frame.copy()

            # 绘制状态信息
            status = "PAUSED - " if self.paused else ""
            status += f"Targets: {self._count_targets()}"
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 绘制正在拖拽的框
            if self.drawing:
                # 获取当前鼠标位置 (通过临时变量)
                pass  # 框在 imshow 之前由 rect 或拖拽状态决定

            # 绘制已确认的框选
            if self.rect:
                x1, y1, x2, y2 = self.rect
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"Press 's' to save ({x2-x1}x{y2-y1})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Live Annotate", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                display_frame = self.frozen_frame if self.paused else self.current_frame
                if display_frame is not None:
                    self.save_target(display_frame)
            elif key == ord('c'):
                self.rect = None
                print("已清除框选")
            elif key == ord('z'):
                self.paused = not self.paused
                if self.paused:
                    self.frozen_frame = self.current_frame.copy() if self.current_frame is not None else None
                    print("已暂停 (方便精确框选)")
                else:
                    self.frozen_frame = None
                    print("已继续")

        cap.release()
        cv2.destroyAllWindows()

    def _count_targets(self):
        info_path = os.path.join(self.targets_dir, "target_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                return len(json.load(f))
        return 0


def main():
    parser = argparse.ArgumentParser(description="实时视频流目标标注工具")
    parser.add_argument("--rtsp", default="rtsp://192.168.144.25:8554/main.264",
                        help="RTSP流地址")
    parser.add_argument("--targets-dir", default=TARGETS_DIR,
                        help="目标模板保存目录")
    parser.add_argument("--recognizer-url", default=None,
                        help="识别引擎HTTP API地址 (如 http://localhost:5000)")
    args = parser.parse_args()

    annotator = LiveAnnotator(
        rtsp_url=args.rtsp,
        targets_dir=args.targets_dir,
        recognizer_url=args.recognizer_url,
    )
    annotator.run()


if __name__ == "__main__":
    main()
