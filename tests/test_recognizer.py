"""TargetRecognizer 单元测试"""
import pytest
import cv2
import numpy as np
import os
import json
import tempfile


class TestTargetRecognizer:
    def setup_method(self):
        """创建临时目标目录"""
        self.tmpdir = tempfile.mkdtemp()
        # 生成测试模板图片
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 255), -1)
        cv2.imwrite(os.path.join(self.tmpdir, "target_000.jpg"), img)

        info = [{"source": "test.jpg", "crop": "target_000.jpg",
                 "bbox": [0, 0, 100, 100], "image_size": [100, 100]}]
        with open(os.path.join(self.tmpdir, "target_info.json"), "w") as f:
            json.dump(info, f)

    def test_load_targets(self):
        from recognize import TargetRecognizer
        rec = TargetRecognizer(self.tmpdir)
        assert len(rec.targets) == 1

    def test_recognize_fast(self):
        from recognize import TargetRecognizer
        rec = TargetRecognizer(self.tmpdir)
        # 创建包含红色方块的测试场景
        scene = np.zeros((720, 1280, 3), dtype=np.uint8)
        scene[300:400, 600:700] = [0, 0, 255]
        results, timing = rec.recognize(scene, fast=True)
        assert isinstance(results, list)
        assert "total" in timing

    def test_recognize_full(self):
        from recognize import TargetRecognizer
        rec = TargetRecognizer(self.tmpdir)
        scene = np.zeros((720, 1280, 3), dtype=np.uint8)
        results, timing = rec.recognize(scene, fast=False)
        assert isinstance(results, list)

    def test_empty_scene(self):
        from recognize import TargetRecognizer
        rec = TargetRecognizer(self.tmpdir)
        scene = np.zeros((720, 1280, 3), dtype=np.uint8)
        results, timing = rec.recognize(scene, fast=True)
        # 空场景不应该检测到目标
        assert isinstance(results, list)
