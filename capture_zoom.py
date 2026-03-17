"""
手动变焦拍摄工具
- 连接 A8mini RTSP流
- 按 + 变焦放大一档，按 - 变焦缩小一档
- 按空格拍照保存
- 按 q 退出

变焦档位: 1x → 2x → 3x → 4x → 6x
"""

import cv2
import os
import threading
from siyi_sdk import SIYIA8mini

ZOOM_LEVELS = [1, 2, 3, 4, 6]
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

cam = SIYIA8mini()

def send_zoom(level):
    threading.Thread(target=lambda: cam.set_zoom(level), daemon=True).start()

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cap = cv2.VideoCapture(cam.rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("无法连接RTSP流")
    exit(1)

zoom_idx = 0
send_zoom(ZOOM_LEVELS[zoom_idx])
count = 0

print("已连接 A8mini")
print(f"变焦档位: {ZOOM_LEVELS}")
print("操作:  +/= 放大  -/_ 缩小  空格=拍照  q=退出")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    info = f"Zoom: {ZOOM_LEVELS[zoom_idx]}x | Photos: {count}"
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
