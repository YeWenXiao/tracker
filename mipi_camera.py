"""
MIPI CSI 视频采集模块 (Jetson 平台)

通过 GStreamer pipeline 使用 nvarguscamerasrc 采集 MIPI CSI 摄像头画面，
利用 Jetson 硬件加速（NVMM内存、nvvidconv）实现低延迟取帧。

典型延迟: ~50ms (对比 RTSP ~1000ms)

用法:
    from mipi_camera import MIPICamera

    cam = MIPICamera(width=1280, height=720, fps=30)
    cam.open()
    frame = cam.read()  # numpy BGR array
    cam.release()
"""

import cv2
import time
import numpy as np


class MIPICamera:
    """Jetson MIPI CSI 摄像头采集器（GStreamer + nvarguscamerasrc）"""

    def __init__(self, sensor_id=0, width=1280, height=720, fps=30,
                 flip_method=0, exposure_range="13000000 358733000",
                 gain_range="1 10"):
        """
        参数:
            sensor_id:      CSI 传感器编号 (0 或 1)
            width:          输出宽度
            height:         输出高度
            fps:            帧率
            flip_method:    翻转方式 (0=不翻转, 2=上下翻转)
            exposure_range: 曝光范围 (ns)，"min max"
            gain_range:     增益范围，"min max"
        """
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.exposure_range = exposure_range
        self.gain_range = gain_range
        self.cap = None
        self._frame_count = 0
        self._start_time = None

    def _build_pipeline(self):
        """构建 GStreamer pipeline 字符串"""
        pipeline = (
            f"nvarguscamerasrc sensor-id={self.sensor_id} "
            f'exposuretimerange="{self.exposure_range}" '
            f'gainrange="{self.gain_range}" ! '
            f"video/x-raw(memory:NVMM),"
            f"width={self.width},height={self.height},"
            f"framerate={self.fps}/1,format=NV12 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw,format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=1"
        )
        return pipeline

    def open(self):
        """打开摄像头"""
        pipeline = self._build_pipeline()
        print(f"[MIPICamera] GStreamer pipeline:")
        print(f"  {pipeline}")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"无法打开 MIPI CSI 摄像头 (sensor_id={self.sensor_id}).\n"
                f"请检查:\n"
                f"  1. CSI 排线连接是否正常\n"
                f"  2. nvarguscamerasrc 是否可用: gst-inspect-1.0 nvarguscamerasrc\n"
                f"  3. OpenCV 是否编译了 GStreamer 支持\n"
                f"Pipeline: {pipeline}"
            )
        self._start_time = time.time()
        self._frame_count = 0
        print(f"[MIPICamera] 已打开 MIPI CSI 摄像头 (sensor={self.sensor_id}, "
              f"{self.width}x{self.height}@{self.fps}fps)")
        return True

    def read(self):
        """
        读取一帧图像

        返回:
            numpy.ndarray (BGR) 或 None (读取失败)
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if ret:
            self._frame_count += 1
            return frame
        return None

    def read_with_timestamp(self):
        """
        读取一帧并返回时间戳

        返回:
            (frame, timestamp_ms) 或 (None, 0)
        """
        if self.cap is None or not self.cap.isOpened():
            return None, 0
        ret, frame = self.cap.read()
        ts = time.time() * 1000  # ms
        if ret:
            self._frame_count += 1
            return frame, ts
        return None, 0

    def is_opened(self):
        """摄像头是否已打开"""
        return self.cap is not None and self.cap.isOpened()

    def get_fps(self):
        """获取实际帧率"""
        if self._start_time is None or self._frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._frame_count / elapsed

    def release(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print(f"[MIPICamera] 已释放 (共采集 {self._frame_count} 帧)")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


def create_capture(source="mipi", rtsp_url="rtsp://192.168.144.25:8554/main.264",
                   width=1280, height=720, fps=30, sensor_id=0):
    """
    统一视频源工厂函数

    参数:
        source:   "mipi" 使用 MIPI CSI, "rtsp" 使用 RTSP 流
        rtsp_url: RTSP 地址 (仅 source="rtsp" 时使用)
        width:    分辨率宽度
        height:   分辨率高度
        fps:      帧率
        sensor_id: MIPI 传感器编号

    返回:
        cv2.VideoCapture 或 MIPICamera 对象 (已打开)
    """
    if source == "mipi":
        cam = MIPICamera(sensor_id=sensor_id, width=width, height=height, fps=fps)
        cam.open()
        return cam
    elif source == "rtsp":
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"无法连接 RTSP: {rtsp_url}")
        print(f"[RTSP] 已连接: {rtsp_url}")
        return cap
    else:
        raise ValueError(f"未知视频源: {source}, 支持 'mipi' 或 'rtsp'")
