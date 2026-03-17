"""
手动变焦拍摄工具
- 默认使用 MIPI CSI 视频源 (低延迟)
- 加 --rtsp 切换为 RTSP 视频源
- 按 + 变焦放大一档，按 - 变焦缩小一档
- 按空格拍照保存
- 按 q 退出

变焦档位: 1x -> 2x -> 3x -> 4x -> 6x
变焦控制始终通过 SIYI SDK (UDP 协议)
"""

import cv2
import os
import argparse
import threading
from siyi_sdk import SIYIA8mini
from mipi_camera import MIPICamera

ZOOM_LEVELS = [1, 2, 3, 4, 6]
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="A8mini 变焦拍摄工具")
    parser.add_argument("--rtsp", action="store_true",
                        help="使用 RTSP 视频源 (默认使用 MIPI CSI)")
    parser.add_argument("--rtsp-url", default="rtsp://192.168.144.25:8554/main.264",
                        help="RTSP 地址 (仅 --rtsp 时使用)")
    args = parser.parse_args()

    cam = SIYIA8mini()  # 云台控制 (变焦始终走 UDP)

    def send_zoom(level):
        threading.Thread(target=lambda: cam.set_zoom(level), daemon=True).start()

    # 打开视频源
    if args.rtsp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("无法连接 RTSP 流")
            exit(1)
        source_name = "RTSP"
    else:
        cap = MIPICamera(width=1280, height=720, fps=30)
        if not cap.isOpened():
            print("无法打开 MIPI CSI 摄像头")
            exit(1)
        source_name = "MIPI CSI"

    zoom_idx = 0
    send_zoom(ZOOM_LEVELS[zoom_idx])
    count = 0

    print(f"已连接 A8mini [{source_name}]")
    print(f"变焦档位: {ZOOM_LEVELS}")
    print("操作:  +/= 放大  -/_ 缩小  空格=拍照  q=退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        info = f"[{source_name}] Zoom: {ZOOM_LEVELS[zoom_idx]}x | Photos: {count}"
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("A8mini", display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('+'), ord('=')):
            if zoom_idx < len(ZOOM_LEVELS) - 1:
                zoom_idx += 1
                send_zoom(ZOOM_LEVELS[zoom_idx])
                print(f"变焦: {ZOOM_LEVELS[zoom_idx]}x")
            else:
                print("已到最大变焦")

        elif key in (ord('-'), ord('_')):
            if zoom_idx > 0:
                zoom_idx -= 1
                send_zoom(ZOOM_LEVELS[zoom_idx])
                print(f"变焦: {ZOOM_LEVELS[zoom_idx]}x")
            else:
                print("已到最小变焦")

        elif key == ord(' '):
            count += 1
            filename = f"zoom_{ZOOM_LEVELS[zoom_idx]}x.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"已保存: {filepath}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cam.close()
    print(f"\n共拍摄 {count} 张照片")


if __name__ == "__main__":
    main()
