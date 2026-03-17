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
        self.total_lost = 0  # 累计丢失帧数
        self.max_lost = 10  # 连续丢失N帧后放弃追踪
        self.tracking = True
        self.track_id = -1  # 由 MultiTracker 分配
        self.age = 0  # 追踪持续帧数
        self.iou_history = []  # IoU 历史 (最近50帧)
        self.trajectory = []  # 轨迹点 (中心坐标)

        # CSRT 追踪器辅助
        self.csrt = cv2.TrackerCSRT_create()
        self.csrt.init(frame, tuple(bbox))

        # 记录初始轨迹点
        x0, y0, w0, h0 = bbox
        self.trajectory.append((int(x0 + w0 / 2), int(y0 + h0 / 2)))

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
        self.age += 1

        # CSRT 追踪
        success, csrt_bbox = self.csrt.update(frame)

        if detection is not None:
            # 有识别结果，用识别结果校正
            x, y, w, h = detection
            cx, cy = x + w / 2, y + h / 2
            self.kalman.correct(np.array([[cx], [cy]], np.float32))
            # 计算 IoU
            iou = self._calc_iou(self.bbox, detection)
            self.iou_history.append(iou)
            if len(self.iou_history) > 50:
                self.iou_history.pop(0)
            self.bbox = detection
            self.lost_count = 0
            # 记录轨迹
            self.trajectory.append((int(cx), int(cy)))
            if len(self.trajectory) > 50:
                self.trajectory.pop(0)
            # 重初始化 CSRT
            self.csrt = cv2.TrackerCSRT_create()
            self.csrt.init(frame, tuple(detection))
            return True, detection
        elif success:
            # CSRT 追踪成功
            bx, by, bw, bh = [int(v) for v in csrt_bbox]
            cx, cy = bx + bw / 2, by + bh / 2
            self.kalman.correct(np.array([[cx], [cy]], np.float32))
            # 计算 IoU
            iou = self._calc_iou(self.bbox, (bx, by, bw, bh))
            self.iou_history.append(iou)
            if len(self.iou_history) > 50:
                self.iou_history.pop(0)
            self.bbox = (bx, by, bw, bh)
            self.lost_count = 0
            # 记录轨迹
            self.trajectory.append((int(cx), int(cy)))
            if len(self.trajectory) > 50:
                self.trajectory.pop(0)
            return True, self.bbox
        else:
            # 追踪丢失
            self.lost_count += 1
            self.total_lost += 1
            if self.lost_count >= self.max_lost:
                self.tracking = False
                log.info("追踪丢失 (连续 %d 帧), 回退全帧识别", self.lost_count)
            pred_bbox = self.predict()
            return False, pred_bbox

    def _calc_iou(self, box1, box2):
        """计算两个 (x,y,w,h) bbox 的 IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def get_quality(self):
        """追踪质量评估"""
        return {
            "track_id": self.track_id,
            "age": self.age,
            "lost_count": self.lost_count,
            "continuity": self.age / max(self.age + self.total_lost, 1),
            "avg_iou": float(np.mean(self.iou_history)) if self.iou_history else 0,
        }

    def is_tracking(self):
        return self.tracking



class MultiTracker:
    """多目标追踪管理器"""

    def __init__(self, max_targets=5):
        self.trackers = {}  # id -> KalmanTracker
        self.next_id = 0
        self.max_targets = max_targets

    def add(self, bbox, frame):
        """添加新追踪目标"""
        if len(self.trackers) >= self.max_targets:
            # 移除最不确定的追踪器 (lost_count 最大的)
            worst = max(self.trackers.items(),
                        key=lambda x: x[1].lost_count)
            del self.trackers[worst[0]]

        tid = self.next_id
        self.next_id += 1
        tracker = KalmanTracker(bbox, frame)
        tracker.track_id = tid
        self.trackers[tid] = tracker
        return tid

    def update(self, frame, detections=None):
        """
        更新所有追踪器
        detections: [(score, x, y, w, h, method), ...]
        返回: {id: (success, bbox)}
        """
        results = {}
        matched_dets = set()

        # 匹配检测结果到已有追踪器 (IoU 匹配)
        if detections:
            for tid, tracker in self.trackers.items():
                best_iou = 0
                best_det = None
                best_idx = -1
                for i, det in enumerate(detections):
                    if i in matched_dets:
                        continue
                    det_bbox = (det[1], det[2], det[3], det[4])
                    iou = self._iou(tracker.bbox, det_bbox)
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_det = det_bbox
                        best_idx = i
                if best_det is not None:
                    matched_dets.add(best_idx)
                    results[tid] = tracker.update(frame, best_det)
                else:
                    results[tid] = tracker.update(frame)

            # 未匹配的检测结果创建新追踪器
            for i, det in enumerate(detections):
                if i not in matched_dets:
                    bbox = (det[1], det[2], det[3], det[4])
                    new_tid = self.add(bbox, frame)
                    results[new_tid] = (True, bbox)
        else:
            for tid, tracker in list(self.trackers.items()):
                results[tid] = tracker.update(frame)

        # 清理丢失的追踪器
        for tid in list(self.trackers.keys()):
            if not self.trackers[tid].is_tracking():
                log.info("多目标追踪: 目标 #%d 已丢失, 移除", tid)
                del self.trackers[tid]
                if tid in results:
                    del results[tid]

        return results

    def _iou(self, box1, box2):
        """计算两个 (x,y,w,h) bbox 的 IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def count(self):
        return len(self.trackers)

    def clear(self):
        self.trackers.clear()

    def get_primary(self):
        """获取主目标 (age 最大的追踪器)"""
        if not self.trackers:
            return None, None
        best_tid = max(self.trackers.keys(),
                       key=lambda t: self.trackers[t].age)
        return best_tid, self.trackers[best_tid]
