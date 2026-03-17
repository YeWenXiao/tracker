"""
MIPI CSI 视频采集模块 (Jetson 平台)

通过 GStreamer pipeline 使用 nvarguscamerasrc 采集 MIPI CSI 摄像头画面，
利用 Jetson 硬件加速（NVMM内存、nvvidconv）实现低延迟取帧。

典型延迟: ~50ms (对比 RTSP ~1000ms)

接口完全兼容 cv2.VideoCapture:
    cam = MIPICamera(width=1280, height=720, fps=30)
    ret, frame = cam.read()       # 与 cv2.VideoCapture.read() 一致
    cam.isOpened()                 # 与 cv2.VideoCapture.isOpened() 一致
    cam.get(cv2.CAP_PROP_...)     # 支持常用属性
    cam.release()
"""

import cv2
import time


class MIPICamera:
    """Jetson MIPI CSI 摄像头采集器（GStreamer + nvarguscamerasrc）

    接口兼容 cv2.VideoCapture，可直接替换使用。
    """

    def __init__(self, sensor_id=0, width=1280, height=720, fps=30,
                 flip_method=0, exposure_range="13000000 358733000",
                 gain_range="1 10"):
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

        # 构造时自动打开 (与 cv2.VideoCapture 行为一致)
        self._open()

    def _build_pipeline(self):
        """构建 GStreamer pipeline 字符串"""
        pipeline = (
            "nvarguscamerasrc sensor-id={sid} "
            'exposuretimerange="{exp}" '
            'gainrange="{gain}" ! '
            "video/x-raw(memory:NVMM),"
            "width={w},height={h},"
            "framerate={fps}/1,format=NV12 ! "
            "nvvidconv flip-method={flip} ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=1"
        ).format(
            sid=self.sensor_id,
            exp=self.exposure_range,
            gain=self.gain_range,
            w=self.width, h=self.height,
            fps=self.fps,
            flip=self.flip_method
        )
        return pipeline

    def _open(self):
        """打开摄像头"""
        pipeline = self._build_pipeline()
        print("[MIPICamera] GStreamer pipeline:")
        print("  " + pipeline)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if self.cap.isOpened():
            self._start_time = time.time()
            self._frame_count = 0
            print("[MIPICamera] 已打开 MIPI CSI 摄像头 (sensor={}, {}x{}@{}fps)".format(
                self.sensor_id, self.width, self.height, self.fps))
        else:
            print("[MIPICamera] 警告: 无法打开 MIPI CSI 摄像头 (sensor_id={})".format(
                self.sensor_id))
            print("  请检查: 1) CSI排线 2) nvarguscamerasrc 3) OpenCV GStreamer支持")

    def isOpened(self):
        """摄像头是否已打开 (兼容 cv2.VideoCapture)"""
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """读取一帧 (兼容 cv2.VideoCapture, 返回 (ret, frame))"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        if ret:
            self._frame_count += 1
        return ret, frame

    def get(self, prop_id):
        """获取属性 (兼容 cv2.VideoCapture)"""
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        elif self.cap is not None:
            return self.cap.get(prop_id)
        return 0.0

    def set(self, prop_id, value):
        """设置属性 (兼容 cv2.VideoCapture, GStreamer 下大多数属性不可写)"""
        if self.cap is not None:
            return self.cap.set(prop_id, value)
        return False

    def get_actual_fps(self):
        """获取实际采集帧率"""
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
            print("[MIPICamera] 已释放 (共采集 {} 帧)".format(self._frame_count))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


def create_capture(source="mipi", rtsp_url="rtsp://192.168.144.25:8554/main.264",
                   width=1280, height=720, fps=30, sensor_id=0):
    """
    统一视频源工厂函数

    返回 cv2.VideoCapture 或 MIPICamera (接口兼容)
    """
    if source == "mipi":
        cam = MIPICamera(sensor_id=sensor_id, width=width, height=height, fps=fps)
        if not cam.isOpened():
            raise RuntimeError("无法打开 MIPI CSI 摄像头")
        return cam
    elif source == "rtsp":
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError("无法连接 RTSP: " + rtsp_url)
        print("[RTSP] 已连接: " + rtsp_url)
        return cap
    else:
        raise ValueError("未知视频源: {}, 支持 'mipi' 或 'rtsp'".format(source))
