"""
多进程流水线架构
- 采集进程：从 MIPI/RTSP 读帧，写入共享内存
- 识别进程：从共享内存读帧，运行识别
- 主进程：显示 + 用户交互

用法:
  python recognize.py --multiprocess --fast    # 多进程快速模式
  python recognize.py --multiprocess           # 多进程全量模式
"""
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import os
import logging

log = logging.getLogger("tracker")


class FrameBuffer:
    """基于共享内存的帧缓冲区"""

    def __init__(self, width=1280, height=720, channels=3, name="tracker_frame"):
        self.shape = (height, width, channels)
        self.nbytes = int(np.prod(self.shape))
        self.name = name
        self.shm = None
        self.timestamp = mp.Value('d', 0.0)
        self.frame_id = mp.Value('i', 0)
        self.lock = mp.Lock()

    def create(self):
        """创建共享内存（采集进程调用）"""
        try:
            # 清理可能残留的同名共享内存
            old = shared_memory.SharedMemory(name=self.name, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        self.shm = shared_memory.SharedMemory(
            name=self.name, create=True, size=self.nbytes)

    def attach(self):
        """附加到已有共享内存（识别/显示进程调用）"""
        self.shm = shared_memory.SharedMemory(name=self.name, create=False)

    def write(self, frame):
        """写入帧"""
        with self.lock:
            arr = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
            np.copyto(arr, frame)
            self.timestamp.value = time.time()
            self.frame_id.value += 1

    def read(self):
        """读取帧，返回 (frame_copy, timestamp, frame_id)"""
        with self.lock:
            arr = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
            frame = arr.copy()
            ts = self.timestamp.value
            fid = self.frame_id.value
        return frame, ts, fid

    def cleanup(self):
        """释放共享内存"""
        if self.shm:
            self.shm.close()
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


def capture_process(frame_buf, source_config, stop_event):
    """
    采集进程: 从视频源读帧写入共享内存

    source_config: dict with keys:
      - type: "mipi" or "rtsp"
      - rtsp_url: RTSP地址 (type=rtsp时)
      - sensor_id, width, height, fps: MIPI参数
    """
    import cv2

    src_type = source_config.get("type", "mipi")
    cap = None

    try:
        frame_buf.attach()

        if src_type == "rtsp":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(source_config["rtsp_url"], cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            log.info("[采集进程] RTSP 已连接: %s", source_config["rtsp_url"])
        else:
            # MIPI CSI
            from mipi_camera import MIPICamera
            cap = MIPICamera(
                sensor_id=source_config.get("sensor_id", 0),
                width=source_config.get("width", 1280),
                height=source_config.get("height", 720),
                fps=source_config.get("fps", 30))
            log.info("[采集进程] MIPI CSI 已连接 (sensor_id=%d)",
                     source_config.get("sensor_id", 0))

        consecutive_failures = 0

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 100:
                    log.error("[采集进程] 视频流持续失败, 退出")
                    break
                time.sleep(0.01)
                continue
            consecutive_failures = 0

            # 确保帧尺寸与共享内存匹配
            h, w = frame.shape[:2]
            exp_h, exp_w = frame_buf.shape[:2]
            if h != exp_h or w != exp_w:
                frame = cv2.resize(frame, (exp_w, exp_h))

            frame_buf.write(frame)

    except Exception as e:
        log.error("[采集进程] 异常: %s", e)
    finally:
        if cap is not None:
            cap.release()
        frame_buf.shm.close()
        log.info("[采集进程] 已退出")


def recognition_process(frame_buf, result_queue, targets_dir, config, stop_event):
    """
    识别进程: 从共享内存读帧运行识别，结果通过Queue发回

    config: dict with keys:
      - fast: bool
      - config_path: str (config.yaml路径)
    """
    import cv2

    try:
        frame_buf.attach()

        # 初始化识别器 (在子进程中)
        from recognize import TargetRecognizer
        rec = TargetRecognizer(targets_dir=targets_dir)
        log.info("[识别进程] 识别器初始化完成, %d 个模板", len(rec.targets))

        fast = config.get("fast", False)
        last_fid = -1

        while not stop_event.is_set():
            frame, ts, fid = frame_buf.read()

            # 跳过已处理的帧
            if fid == last_fid:
                time.sleep(0.005)
                continue
            last_fid = fid

            # 跳过过时的帧 (>500ms)
            if ts > 0 and time.time() - ts > 0.5:
                time.sleep(0.005)
                continue

            # 识别
            results, timing = rec.recognize(frame, fast=fast)

            # 非阻塞发送结果
            try:
                # 清空旧结果，只保留最新
                while not result_queue.empty():
                    try:
                        result_queue.get_nowait()
                    except Exception:
                        break
                result_queue.put_nowait({
                    "results": results,
                    "timing": timing,
                    "frame_id": fid,
                    "timestamp": ts,
                })
            except Exception:
                pass

    except Exception as e:
        log.error("[识别进程] 异常: %s", e)
    finally:
        frame_buf.shm.close()
        log.info("[识别进程] 已退出")


def run_multiprocess(args, cfg, rec_instance=None):
    """
    启动多进程流水线

    args: argparse.Namespace (命令行参数)
    cfg: dict (config.yaml配置)
    rec_instance: 可选，主进程中已初始化的识别器（用于模板信息）
    """
    import cv2
    from recognize import draw_results, get_system_stats, FPSCounter

    width = args.width
    height = args.height

    # 创建共享帧缓冲
    frame_buf = FrameBuffer(width=width, height=height, name="tracker_frame")
    frame_buf.create()

    # 创建结果队列
    result_queue = mp.Queue(maxsize=2)
    stop_event = mp.Event()

    # 视频源配置
    source_config = {
        "type": "rtsp" if args.rtsp else "mipi",
        "rtsp_url": args.rtsp_url,
        "sensor_id": args.sensor_id,
        "width": width,
        "height": height,
        "fps": args.fps_cap,
    }

    # 识别配置
    recog_cfg = cfg.get("recognition", {})
    recog_config = {
        "fast": args.fast,
        "config_path": "config.yaml",
    }
    targets_dir = recog_cfg.get("targets_dir", "targets")

    # 启动子进程
    cap_proc = mp.Process(
        target=capture_process,
        args=(frame_buf, source_config, stop_event),
        name="capture")
    rec_proc = mp.Process(
        target=recognition_process,
        args=(frame_buf, result_queue, targets_dir, recog_config, stop_event),
        name="recognition")

    cap_proc.start()
    rec_proc.start()
    log.info("[主进程] 多进程流水线已启动 (采集PID=%d, 识别PID=%d)",
             cap_proc.pid, rec_proc.pid)

    # 主进程: 显示循环
    cv2.namedWindow("A8mini Live [MP]")
    fps_counter = FPSCounter(window_size=30)
    latest_results = []
    latest_timing = {}
    sys_stats_cache = {}
    sys_stats_time = 0
    last_fid = -1

    try:
        while True:
            # 从共享内存读帧
            frame, ts, fid = frame_buf.read()
            if fid == last_fid:
                time.sleep(0.005)
                continue
            last_fid = fid

            # 从队列读最新识别结果
            try:
                while not result_queue.empty():
                    data = result_queue.get_nowait()
                    latest_results = data["results"]
                    latest_timing = data["timing"]
            except Exception:
                pass

            current_fps = fps_counter.tick()

            # 系统状态
            now = time.time()
            if now - sys_stats_time > 1.0:
                sys_stats_cache = get_system_stats()
                sys_stats_time = now

            source_label = "RTSP" if args.rtsp else "MIPI"
            mode_str = "FAST" if args.fast else "FULL"
            status = f"{source_label} {mode_str} [MP]"

            display = draw_results(frame, latest_results, latest_timing,
                                   status, fps=current_fps,
                                   system_stats=sys_stats_cache)
            cv2.imshow("A8mini Live [MP]", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            # 检查子进程存活
            if not cap_proc.is_alive() or not rec_proc.is_alive():
                log.warning("[主进程] 子进程异常退出")
                break

    except KeyboardInterrupt:
        log.info("[主进程] Ctrl+C 退出")
    finally:
        stop_event.set()
        cap_proc.join(timeout=3)
        rec_proc.join(timeout=3)
        if cap_proc.is_alive():
            cap_proc.terminate()
        if rec_proc.is_alive():
            rec_proc.terminate()
        frame_buf.cleanup()
        cv2.destroyAllWindows()
        log.info("[主进程] 多进程流水线已关闭")
