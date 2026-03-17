"""KalmanTracker 单元测试"""
import pytest
import cv2
import numpy as np


class TestKalmanTracker:
    def test_init(self):
        from tracker import KalmanTracker
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        t = KalmanTracker((100, 100, 50, 50), frame)
        assert t.is_tracking()

    def test_predict(self):
        from tracker import KalmanTracker
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        t = KalmanTracker((100, 100, 50, 50), frame)
        bbox = t.predict()
        assert len(bbox) == 4

    def test_update_with_detection(self):
        from tracker import KalmanTracker
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        t = KalmanTracker((100, 100, 50, 50), frame)
        success, bbox = t.update(frame, detection=(110, 110, 50, 50))
        assert success

    def test_lost_tracking(self):
        from tracker import KalmanTracker
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        t = KalmanTracker((100, 100, 50, 50), frame)
        # 模拟连续丢失
        for _ in range(15):
            t.update(frame)
        assert not t.is_tracking()
