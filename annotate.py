"""
目标标注工具
- 加载 captures/ 中的照片
- 鼠标拖框选中目标
- 按 s 保存，按 n/p 切换，按 q 退出
- 裁剪保存到 targets/，生成 target_info.json
"""

import cv2
import os
import json
import glob

CAPTURES_DIR = "captures"
TARGETS_DIR = "targets"
os.makedirs(TARGETS_DIR, exist_ok=True)

images = sorted(glob.glob(os.path.join(CAPTURES_DIR, "*.jpg")))
if not images:
    print("captures/ 中没有图片")
    exit(1)

print(f"找到 {len(images)} 张图片")
print("操作: 鼠标拖框 → s=保存 → n=下一张 → p=上一张 → q=退出")

drawing = False
ix, iy = 0, 0
rect = None
current_frame = None


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, rect, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        disp = current_frame.copy()
        cv2.rectangle(disp, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Annotate", disp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        if x2 - x1 > 10 and y2 - y1 > 10:
            rect = (x1, y1, x2, y2)
            disp = current_frame.copy()
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp, "Press 's' to save", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Annotate", disp)


cv2.namedWindow("Annotate")
cv2.setMouseCallback("Annotate", mouse_cb)

idx = 0
annotations = []

while True:
    current_frame = cv2.imread(images[idx])
    h, w = current_frame.shape[:2]
    rect = None
    disp = current_frame.copy()
    cv2.putText(disp, f"[{idx+1}/{len(images)}] {os.path.basename(images[idx])}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Annotate", disp)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('s') and rect is not None:
        x1, y1, x2, y2 = rect
        crop = current_frame[y1:y2, x1:x2]
        crop_name = f"target_{len(annotations):03d}.jpg"
        crop_path = os.path.join(TARGETS_DIR, crop_name)
        cv2.imwrite(crop_path, crop)
        annotations.append({
            "source": os.path.basename(images[idx]),
            "crop": crop_name,
            "bbox": [x1, y1, x2, y2],
            "image_size": [w, h]
        })
        print(f"已保存: {crop_path} ({x2-x1}x{y2-y1})")
        idx = min(idx + 1, len(images) - 1)
    elif key == ord('n'):
        idx = min(idx + 1, len(images) - 1)
    elif key == ord('p'):
        idx = max(idx - 1, 0)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

if annotations:
    info_path = os.path.join(TARGETS_DIR, "target_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"\n标注完成! 共 {len(annotations)} 个, 保存到 {TARGETS_DIR}/")
else:
    print("\n未保存任何标注")
