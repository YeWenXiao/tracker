"""
RTSP Live Target Annotation Tool
- Connect to RTSP stream
- Mouse drag to select region
- Press s to save crop to targets/, auto-update target_info.json
- Press q to quit
- After save, auto-triggers recognize.py hot-reload (via file monitor or HTTP API)

Usage:
  python annotate_live.py
  python annotate_live.py --rtsp rtsp://192.168.144.25:8554/main.264
  python annotate_live.py --api http://localhost:5000
"""

import cv2
import os
import json
import argparse
import time

TARGETS_DIR = "targets"
INFO_FILE = "target_info.json"

drawing = False
ix, iy = 0, 0
rect = None
current_frame = None


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        if x2 - x1 > 10 and y2 - y1 > 10:
            rect = (x1, y1, x2, y2)


def save_crop(frame, rect_coords, api_url=None):
    os.makedirs(TARGETS_DIR, exist_ok=True)

    info_path = os.path.join(TARGETS_DIR, INFO_FILE)
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    else:
        annotations = []

    x1, y1, x2, y2 = rect_coords
    h, w = frame.shape[:2]
    crop = frame[y1:y2, x1:x2]

    idx = len(annotations)
    crop_name = f"target_{idx:03d}.jpg"
    crop_path = os.path.join(TARGETS_DIR, crop_name)
    while os.path.exists(crop_path):
        idx += 1
        crop_name = f"target_{idx:03d}.jpg"
        crop_path = os.path.join(TARGETS_DIR, crop_name)

    cv2.imwrite(crop_path, crop)

    annotations.append({
        "source": "live_annotate",
        "crop": crop_name,
        "bbox": [x1, y1, x2, y2],
        "image_size": [w, h]
    })

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"Saved: {crop_path} ({x2-x1}x{y2-y1})")

    if api_url:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{api_url}/api/targets/reload",
                method="POST",
                data=b"",
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=2)
            print("Reload triggered via HTTP API")
        except Exception as e:
            print(f"HTTP API trigger failed (file monitor will still work): {e}")
    else:
        print("Target saved, file monitor will auto-trigger reload")

    return crop_name


def main():
    global current_frame, rect, drawing, ix, iy

    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", default="rtsp://192.168.144.25:8554/main.264")
    parser.add_argument("--api", default=None, help="HTTP API URL (e.g. http://localhost:5000)")
    args = parser.parse_args()

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    print(f"Connecting RTSP: {args.rtsp}")
    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Cannot connect to RTSP stream")
        return

    cv2.namedWindow("Live Annotate")
    cv2.setMouseCallback("Live Annotate", mouse_cb)

    print("Controls: mouse drag -> s=save target -> q=quit")
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        current_frame = frame.copy()
        display = frame.copy()

        if rect is not None:
            x1, y1, x2, y2 = rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "Press \'s\' to save", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display, f"Saved: {saved_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Annotate", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and rect is not None:
            crop_name = save_crop(current_frame, rect, args.api)
            saved_count += 1
            rect = None
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total saved: {saved_count} targets")


if __name__ == "__main__":
    main()
