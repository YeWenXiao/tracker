"""
YOLO 检测器 — 同步模式(Orin Nano够快)，保留异步备选
"""

import threading
import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH, YOLO_CONF, YOLO_IMGSZ


class Detector:
    """YOLO目标检测器"""

    def __init__(self, model_path=None, conf=None, imgsz=None):
        self.model_path = model_path or MODEL_PATH
        self.conf = conf or YOLO_CONF
        self.imgsz = imgsz or YOLO_IMGSZ
        self.model = None

    def load(self):
        print(f'[检测] 加载模型: {self.model_path}')
        self.model = YOLO(self.model_path)
        print(f'[检测] 已加载, 置信度: {self.conf}')

    def detect(self, frame):
        """
        检测一帧，返回 [(x1,y1,x2,y2,conf,cls), ...] 或空列表
        """
        if self.model is None:
            return []

        results = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False
        )

        detections = []
        if results and results[0].boxes and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clses = results[0].boxes.cls.cpu().numpy()
            for i in range(len(confs)):
                x1, y1, x2, y2 = boxes[i]
                detections.append((
                    int(x1), int(y1), int(x2), int(y2),
                    float(confs[i]), int(clses[i])
                ))

        return detections

    def detect_crops(self, frame, detections):
        """从检测框中裁剪目标区域"""
        crops = []
        for x1, y1, x2, y2, conf, cls in detections:
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                crops.append(crop)
            else:
                crops.append(None)
        return crops
