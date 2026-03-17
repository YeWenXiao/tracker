"""
卡尔曼滤波目标追踪器
- 识别到目标后启动追踪
- 预测下一帧目标位置
- 追踪丢失后回退到全帧识别
"""
import cv2
import numpy as np
import logging

log = logging.getLogger("tracker.kalman")


class KalmanTracker:
    """基于 OpenCV 卡尔曼滤波的目标追踪"""

    def __init__(self, bbox, frame):
        """
        初始化追踪器
        bbox: (x, y, w, h)
        frame: 初始帧
        """
        self.kalman = cv2.KalmanFilter(4, 2)  # 状态: x,y,vx,vy  观测: x,y
        # 状态转移矩阵 (匀速运动模型)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        # 初始状态
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

        self.bbox = bbox
        self.lost_count = 0
        self.max_lost = 10  # 连续丢失N帧后放弃追踪
        self.tracking = True

        # CSRT 追踪器辅助
        self.csrt = cv2.TrackerCSRT_create()
        self.csrt.init(frame, tuple(bbox))

    def predict(self):
        """预测下一帧位置"""
        pred = self.kalman.predict()
        cx, cy = pred[0, 0], pred[1, 0]
        x, y, w, h = self.bbox
        return (int(cx - w / 2), int(cy - h / 2), w, h)

    def update(self, frame, detection=None):
        """
        更新追踪状态
        detection: 识别结果 (x, y, w, h) 或 None
        返回: (success, bbox)
        """
        # CSRT 追踪
        success, csrt_bbox = self.csrt.update(frame)

        if detection is not None:
            # 有识别结果，用识别结果校正
            x, y, w, h = detection
            cx, cy = x + w / 2, y + h / 2
            self.kalman.correct(np.array([[cx], [cy]], np.float32))
            self.bbox = detection
            self.lost_count = 0
            # 重初始化 CSRT
            self.csrt = cv2.TrackerCSRT_create()
            self.csrt.init(frame, tuple(detection))
            return True, detection
        elif success:
            # CSRT 追踪成功
            bx, by, bw, bh = [int(v) for v in csrt_bbox]
            cx, cy = bx + bw / 2, by + bh / 2
            self.kalman.correct(np.array([[cx], [cy]], np.float32))
            self.bbox = (bx, by, bw, bh)
            self.lost_count = 0
            return True, self.bbox
        else:
            # 追踪丢失
            self.lost_count += 1
            if self.lost_count >= self.max_lost:
                self.tracking = False
                log.info("追踪丢失 (连续 %d 帧), 回退全帧识别", self.lost_count)
            pred_bbox = self.predict()
            return False, pred_bbox

    def is_tracking(self):
        return self.tracking
