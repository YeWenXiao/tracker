"""
RTSP 视频流读取 — 从v1.5搬来
"""

import os
import threading
import time
import cv2

# RTSP必须用TCP传输，UDP在Windows上丢帧严重
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'


class RTSPReader:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        for _ in range(50):
            time.sleep(0.1)
            if self.frame is not None:
                return True
        return False

    def _loop(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f'[RTSP] 无法连接: {self.url}')
            self.running = False
            return
        print(f'[RTSP] 已连接')
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)
        cap.release()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
