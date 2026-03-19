"""
弹载视觉追踪系统 V2.0 — 三状态机主循环

工作流程:
  1. 启动时从照片集预提取目标特征
  2. SEARCH: 云台扫描 + YOLO粗筛 + 特征匹配确认身份
  3. TRACK:  PID居中 + zoom随接近自动递减 + 持续特征验证
  4. TERMINAL: 目标占画面>50%，最高精度锁定

用法:
  python main.py                              # 默认配置
  python main.py --photos target_photos/      # 指定照片集
  python main.py --no_gimbal                  # 无云台模式(纯检测)
  python main.py --source video.mp4           # 用视频文件测试
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

from config import *
from gimbal import SIYIGimbal
from pid import PID
from video import RTSPReader
from detector import Detector
from feature_bank import TargetFeatureBank
from target_matcher import TargetMatcher
from zoom_manager import ZoomManager
from web import WebServer


def main():
    has_display = os.environ.get('DISPLAY') is not None or sys.platform == 'win32'

    parser = argparse.ArgumentParser(description='V2.0 弹载视觉追踪')
    parser.add_argument('--camera_ip', default=CAMERA_IP)
    parser.add_argument('--rtsp_url', default=None)
    parser.add_argument('--source', default=None, help='视频文件(测试用)')
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--conf', type=float, default=YOLO_CONF)
    parser.add_argument('--photos', default='target_photos', help='目标照片集目录')
    parser.add_argument('--web', type=int, default=WEB_PORT)
    parser.add_argument('--no_gimbal', action='store_true')
    parser.add_argument('--no_zoom', action='store_true')
    args = parser.parse_args()

    if args.rtsp_url is None:
        args.rtsp_url = f'rtsp://{args.camera_ip}:8554/main.264'

    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)

    photos_dir = args.photos
    if not os.path.isabs(photos_dir):
        photos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), photos_dir)

    print('=' * 50)
    print('  弹载视觉追踪系统 V2.0')
    print('  照片集特征匹配 + 三状态机追踪')
    print('=' * 50)

    # ==================== 1. 加载目标特征库 ====================
    feature_bank = TargetFeatureBank()
    if os.path.exists(photos_dir):
        feature_bank.load_from_dir(photos_dir)
    else:
        print(f'[警告] 照片集目录不存在: {photos_dir}')
        print(f'[警告] 将仅使用YOLO检测，无特征匹配验证')

    matcher = TargetMatcher(feature_bank)

    # ==================== 2. 加载YOLO模型 ====================
    detector = Detector(model_path=model_path, conf=args.conf)
    detector.load()

    # ==================== 3. 连接云台 ====================
    gimbal = None
    if not args.no_gimbal:
        gimbal = SIYIGimbal(args.camera_ip)
        gimbal.center()
        # 强制zoom回1x
        gimbal.zoom_out()
        time.sleep(3)
        gimbal.zoom_stop()
        time.sleep(1)
        print(f'[云台] 已连接并回正: {args.camera_ip}, zoom=1x')

    # ==================== 4. PID ====================
    yaw_pid = PID(kp=25, ki=0.05, kd=5.0, max_out=60)
    pitch_pid = PID(kp=25, ki=0.05, kd=5.0, max_out=60)

    # ==================== 5. Zoom管理 ====================
    zoom_mgr = ZoomManager(gimbal)
    zoom_mgr.enabled = not args.no_zoom

    # ==================== 6. 视频源 ====================
    cap = None
    reader = None
    if args.source:
        # 视频文件模式（测试用）
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f'[视频] 无法打开: {args.source}')
            sys.exit(1)
        print(f'[视频] 已打开文件: {args.source}')
    else:
        print(f'[RTSP] 连接: {args.rtsp_url}')
        reader = RTSPReader(args.rtsp_url)
        if not reader.start():
            print('[RTSP] 连接失败!')
            sys.exit(1)

    # ==================== 7. Web服务 ====================
    web = WebServer(args.web)

    # ==================== 状态变量 ====================
    state = STATE_SEARCH
    search_start_time = time.time()
    track_box = None            # 当前追踪目标 (x1,y1,x2,y2)
    track_conf = 0.0            # 当前目标置信度
    track_match_score = 0.0     # 当前目标匹配分
    lost_start_time = 0         # 丢失开始时间
    last_lock_angle = None      # 最后锁定的云台角度
    last_verify_time = 0        # 上次特征验证时间

    # 颜色搜索节流
    last_color_search_time = 0
    COLOR_SEARCH_INTERVAL = 0.15  # 搜索阶段加快颜色搜索频率

    # 扫描（延迟启动，先静止检测）
    SCAN_DELAY = 2.0            # 前2秒不动云台，先在当前画面找
    scan_direction = 1
    scan_last_switch = time.time()
    scan_pitch_idx = 0
    scan_sweep_count = 0

    # FPS
    fps = 0.0
    fps_count = 0
    fps_time = time.time()
    last_time = time.time()

    # 录像 — 带时间戳保存到桌面
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(os.path.expanduser('~/Desktop'), f'tracker_test_{timestamp}.avi')
    video_writer = None

    def read_frame():
        if cap is not None:
            ret, frame = cap.read()
            return frame if ret else None
        elif reader is not None:
            return reader.read()
        return None

    def do_center():
        nonlocal state, track_box, lost_start_time
        state = STATE_SEARCH
        track_box = None
        lost_start_time = 0
        if gimbal:
            gimbal.stop()
            gimbal.center()
        zoom_mgr.zoom_to_min()
        yaw_pid.reset()
        pitch_pid.reset()
        return '云台回中，重新搜索'

    def do_rescan():
        nonlocal state, track_box, lost_start_time, search_start_time
        state = STATE_SEARCH
        track_box = None
        lost_start_time = 0
        search_start_time = time.time()
        if gimbal:
            gimbal.stop()
        zoom_mgr.zoom_to_min()
        yaw_pid.reset()
        pitch_pid.reset()
        return '重新搜索'

    def do_zoom_in():
        if gimbal:
            gimbal.zoom_in()
        return 'Zoom+'

    def do_zoom_out():
        if gimbal:
            gimbal.zoom_out()
        return 'Zoom-'

    web.callbacks = {
        'center': do_center,
        'rescan': do_rescan,
        'zoom_in': do_zoom_in,
        'zoom_out': do_zoom_out,
    }
    web.start()
    print(f'[Web] http://0.0.0.0:{args.web}/')
    print(f'\n[系统] V2.0 运行中... 按 Ctrl+C 退出\n')

    try:
        while True:
            frame = read_frame()
            if frame is None:
                if cap is not None:
                    break  # 视频播放完毕
                time.sleep(0.01)
                continue

            now = time.time()
            dt = now - last_time
            last_time = now

            # FPS
            fps_count += 1
            if now - fps_time >= 1.0:
                fps = fps_count / (now - fps_time)
                fps_count = 0
                fps_time = now

            h, w = frame.shape[:2]

            # ==================== YOLO检测 ====================
            detections = detector.detect(frame)

            # 过滤太大(>40%)或太小(<15px)的框
            detections = [d for d in detections
                          if max((d[2]-d[0])/w, (d[3]-d[1])/h) < 0.40
                          and (d[2]-d[0]) > 15 and (d[3]-d[1]) > 15]

            # ==================== 状态机 ====================

            if state == STATE_SEARCH:
                # ========== 搜索阶段 ==========
                elapsed = now - search_start_time

                # 先检测，找到就立刻切追踪，不发扫描指令
                found_det = None
                found_score = 0.0

                if feature_bank.is_loaded():
                    # 1. 颜色搜索
                    color_candidates = []
                    if now - last_color_search_time > COLOR_SEARCH_INTERVAL:
                        color_candidates = matcher.color_search(frame)
                        last_color_search_time = now
                    if color_candidates and color_candidates[0][4] > COLOR_SEARCH_THRESHOLD:
                        best = color_candidates[0]
                        found_det = (best[0], best[1], best[2], best[3], best[4], 0)
                        found_score = best[4]

                    # 2. 如果颜色搜索没找到，再试YOLO+特征匹配
                    if found_det is None and detections:
                        found_det, found_score = matcher.match_detections(frame, detections)

                elif detections:
                    best = max(detections, key=lambda d: d[4])
                    if best[4] > args.conf:
                        found_det = best
                        found_score = best[4]

                if found_det is not None:
                    # 找到目标! → 立刻停止云台并进入追踪
                    if gimbal:
                        gimbal.stop()
                    track_box = found_det[:4]
                    track_conf = found_det[4]
                    track_match_score = found_score
                    lost_start_time = 0
                    yaw_pid.reset()
                    pitch_pid.reset()
                    state = STATE_TRACK
                    print(f'[搜索→追踪] 找到目标! conf:{track_conf:.2f} match:{found_score:.2f} 耗时:{elapsed:.1f}s')
                else:
                    # 没找到才扫描（前SCAN_DELAY秒不动）
                    if gimbal and elapsed > SCAN_DELAY:
                        if now - scan_last_switch > SCAN_STEP_TIME:
                            scan_direction *= -1
                            scan_sweep_count += 1
                            scan_last_switch = now
                            if scan_sweep_count % 2 == 0:
                                scan_pitch_idx = (scan_pitch_idx + 1) % len(SCAN_PITCH_ANGLES)
                                gimbal.set_angle(0, SCAN_PITCH_ANGLES[scan_pitch_idx])
                        gimbal.set_speed(SCAN_SPEED * scan_direction, 0)

                    zoom_mgr.set_search_zoom(elapsed, SEARCH_ZOOM_SCHEDULE)

            elif state == STATE_TRACK:
                # ========== 追踪阶段 ==========
                x1, y1, x2, y2 = track_box if track_box else (0, 0, 0, 0)
                box_ratio = max((x2-x1)/w, (y2-y1)/h) if track_box else 0

                # 找到当前目标: 优先从检测中选距离上次最近的
                matched_det = None
                if detections and track_box:
                    last_cx = (x1 + x2) / 2
                    last_cy = (y1 + y2) / 2
                    last_size = max(x2 - x1, y2 - y1)
                    max_dist = max(last_size * 3, 150)

                    # 如果有特征库，用特征匹配；否则用距离+置信度
                    if feature_bank.is_loaded():
                        matched_det, match_score = matcher.match_detections(
                            frame, detections, threshold=VERIFY_THRESHOLD
                        )
                        if matched_det:
                            # 还要检查距离
                            mcx = (matched_det[0] + matched_det[2]) / 2
                            mcy = (matched_det[1] + matched_det[3]) / 2
                            dist = ((mcx - last_cx)**2 + (mcy - last_cy)**2)**0.5
                            if dist > max_dist:
                                matched_det = None
                    else:
                        best_idx = None
                        best_score = 0
                        for det in detections:
                            dcx = (det[0] + det[2]) / 2
                            dcy = (det[1] + det[3]) / 2
                            dist = ((dcx - last_cx)**2 + (dcy - last_cy)**2)**0.5
                            if dist < max_dist and det[4] > best_score:
                                best_score = det[4]
                                matched_det = det

                # YOLO没匹配到时，用颜色搜索补救
                if matched_det is None and feature_bank.is_loaded() and track_box:
                    color_candidates = matcher.color_search(frame)
                    if color_candidates:
                        last_cx = (track_box[0] + track_box[2]) / 2
                        last_cy = (track_box[1] + track_box[3]) / 2
                        last_size = max(track_box[2] - track_box[0], track_box[3] - track_box[1])
                        max_dist = max(last_size * 3, 150)
                        for cand in color_candidates:
                            cx = (cand[0] + cand[2]) / 2
                            cy = (cand[1] + cand[3]) / 2
                            dist = ((cx - last_cx)**2 + (cy - last_cy)**2)**0.5
                            if dist < max_dist and cand[4] > COLOR_SEARCH_THRESHOLD:
                                matched_det = (cand[0], cand[1], cand[2], cand[3], cand[4], 0)
                                break

                if matched_det:
                    # === 目标可见 ===
                    x1, y1, x2, y2 = matched_det[:4]
                    track_box = (x1, y1, x2, y2)
                    track_conf = matched_det[4]
                    lost_start_time = 0
                    box_ratio = max((x2-x1)/w, (y2-y1)/h)

                    # 自适应PID参数
                    yaw_pid.set_profile(box_ratio)
                    pitch_pid.set_profile(box_ratio)

                    # PID居中
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    err_x = (cx - w / 2) / (w / 2)
                    err_y = (cy - h / 2) / (h / 2)

                    yaw_speed = yaw_pid.compute(err_x, dt)
                    pitch_speed = pitch_pid.compute(err_y, dt)

                    if gimbal:
                        gimbal.set_speed(yaw_speed, -pitch_speed)

                    # Zoom放大（只有目标接近中心才放大）
                    center_error = max(abs(err_x), abs(err_y))
                    zoom_mgr.update(box_ratio, center_error)

                    # 持续特征验证（每秒一次）
                    if (feature_bank.is_loaded() and
                            now - last_verify_time > VERIFY_INTERVAL):
                        crop = frame[max(0,y1):y2, max(0,x1):x2]
                        if crop.size > 0:
                            verified = matcher.verify_target(crop)
                            last_verify_time = now
                            if not verified:
                                print(f'[追踪] 特征验证失败! 跟错目标，回搜索')
                                state = STATE_SEARCH
                                search_start_time = now
                                track_box = None
                                lost_start_time = 0
                                if gimbal:
                                    gimbal.stop()
                                yaw_pid.reset()
                                pitch_pid.reset()

                    # 目标够大后不切状态，继续PID居中保持锁定

                else:
                    # === 目标丢失 ===
                    if lost_start_time == 0:
                        lost_start_time = now
                        print(f'[追踪] 目标丢失! 回到上次位置搜索')
                        zoom_mgr.stop()

                    lost_duration = now - lost_start_time

                    if lost_duration < LOST_SPIRAL_TIME:
                        # 回到上次锁定角度，小范围搜索
                        if gimbal and track_box:
                            # 轻微摆动搜索
                            sweep = LOST_SPIRAL_RANGE * np.sin(lost_duration * 2.0)
                            gimbal.set_speed(sweep, 0)
                    elif lost_duration < LOST_TIMEOUT_SEARCH:
                        # 扩大搜索范围
                        if gimbal:
                            sweep = LOST_SPIRAL_RANGE * 2 * np.sin(lost_duration * 1.5)
                            gimbal.set_speed(sweep, 0)
                    else:
                        # 超时 → 回到搜索
                        print(f'[追踪→搜索] 丢失超时({lost_duration:.1f}s)，全局搜索')
                        state = STATE_SEARCH
                        search_start_time = now
                        track_box = None
                        lost_start_time = 0
                        zoom_mgr.zoom_to_min()
                        yaw_pid.reset()
                        pitch_pid.reset()

            elif state == STATE_TERMINAL:
                # ========== 末端阶段 ==========
                # 最高精度PID锁定，zoom回1x
                zoom_mgr.update(1.0)  # 强制zoom=1

                matched_det = None
                if detections and track_box:
                    x1, y1, x2, y2 = track_box
                    last_cx = (x1 + x2) / 2
                    last_cy = (y1 + y2) / 2
                    # 末端阶段只找最近的大目标
                    best_dist = 9999
                    for det in detections:
                        dcx = (det[0] + det[2]) / 2
                        dcy = (det[1] + det[3]) / 2
                        dist = ((dcx - last_cx)**2 + (dcy - last_cy)**2)**0.5
                        if dist < best_dist:
                            best_dist = dist
                            matched_det = det

                if matched_det:
                    x1, y1, x2, y2 = matched_det[:4]
                    track_box = (x1, y1, x2, y2)
                    track_conf = matched_det[4]
                    box_ratio = max((x2-x1)/w, (y2-y1)/h)

                    # 末端PID参数
                    yaw_pid.set_profile(box_ratio)
                    pitch_pid.set_profile(box_ratio)

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    err_x = (cx - w / 2) / (w / 2)
                    err_y = (cy - h / 2) / (h / 2)

                    yaw_speed = yaw_pid.compute(err_x, dt)
                    pitch_speed = pitch_pid.compute(err_y, dt)

                    if gimbal:
                        gimbal.set_speed(yaw_speed, -pitch_speed)

                    # 如果目标缩小了（距离拉远），退回追踪
                    if box_ratio < TERMINAL_BOX_RATIO * 0.7:
                        state = STATE_TRACK
                        print(f'[末端→追踪] 目标缩小({box_ratio:.0%})，退回追踪')
                else:
                    # 末端丢失 → 直接回追踪(丢失恢复)
                    if lost_start_time == 0:
                        lost_start_time = now
                    if now - lost_start_time > 2.0:
                        state = STATE_TRACK
                        print(f'[末端→追踪] 目标丢失，退回追踪恢复')

            # ==================== OSD绘制 ====================
            vis = frame.copy()

            # 十字准星
            cs = 25
            cv2.line(vis, (w//2 - cs, h//2), (w//2 + cs, h//2), (0, 0, 255), 2)
            cv2.line(vis, (w//2, h//2 - cs), (w//2, h//2 + cs), (0, 0, 255), 2)

            # 所有YOLO检测框（灰色细框）
            for det in detections:
                dx1, dy1, dx2, dy2, dconf, dcls = det
                cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), (128, 128, 128), 1)

            # 当前追踪目标
            if track_box and state in (STATE_TRACK, STATE_TERMINAL):
                x1, y1, x2, y2 = track_box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                box_ratio = max((x2-x1)/w, (y2-y1)/h)

                # 颜色: 追踪=绿色, 末端=红色, 丢失=黄色
                if state == STATE_TERMINAL:
                    color = (0, 0, 255)
                elif lost_start_time > 0:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.line(vis, (w//2, h//2), (cx, cy), (0, 255, 255), 1)

                # 信息标签
                label = f'conf:{track_conf:.2f} match:{track_match_score:.2f} size:{box_ratio:.0%}'
                cv2.putText(vis, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 状态栏
            status = f'{state}'
            if state == STATE_TRACK and lost_start_time > 0:
                status += f' LOST:{now - lost_start_time:.1f}s'
            status += f' zoom:{zoom_mgr.current_zoom}x'

            status_color = {
                STATE_SEARCH: (200, 200, 200),
                STATE_TRACK: (0, 255, 0),
                STATE_TERMINAL: (0, 0, 255),
            }.get(state, (200, 200, 200))

            if lost_start_time > 0 and state == STATE_TRACK:
                status_color = (0, 165, 255)

            cv2.putText(vis, f'FPS:{fps:.0f} | {status}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # 特征库状态
            if feature_bank.is_loaded():
                cv2.putText(vis, f'Templates:{len(feature_bank.templates)}', (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

            web.update_frame(vis)
            web.status_text = f'FPS:{fps:.0f} | {status}'

            # 录像
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(save_path, fourcc, 15.0, (w, h))
            video_writer.write(vis)

            # 本地显示
            if has_display:
                if fps_count <= 1:
                    cv2.namedWindow('V2.0 Tracker', cv2.WINDOW_NORMAL)
                cv2.imshow('V2.0 Tracker', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # 每秒打印一次状态
            if fps_count == 0:
                if state == STATE_SEARCH:
                    elapsed = now - search_start_time
                    print(f'[搜索] {elapsed:.0f}s zoom:{zoom_mgr.current_zoom}x det:{len(detections)}')
                elif state == STATE_TRACK and track_box:
                    bx1, by1, bx2, by2 = track_box
                    br = max((bx2-bx1)/w, (by2-by1)/h)
                    lost_str = f' LOST:{now-lost_start_time:.1f}s' if lost_start_time > 0 else ''
                    print(f'[追踪] conf:{track_conf:.2f} match:{track_match_score:.2f} '
                          f'size:{br:.0%} zoom:{zoom_mgr.current_zoom}x{lost_str}')
                elif state == STATE_TERMINAL and track_box:
                    bx1, by1, bx2, by2 = track_box
                    br = max((bx2-bx1)/w, (by2-by1)/h)
                    print(f'[末端] size:{br:.0%} 精确锁定中')

    except KeyboardInterrupt:
        print('\n[系统] 用户退出')
    finally:
        if video_writer:
            video_writer.release()
            print(f'[录像] 已保存: {save_path}')
        if reader:
            reader.stop()
        if cap:
            cap.release()
        if gimbal:
            gimbal.stop()
            gimbal.zoom_stop()
            gimbal.close()
        zoom_mgr.stop()
        web.stop()
        if has_display:
            cv2.destroyAllWindows()
        print('[系统] V2.0 已退出')


if __name__ == '__main__':
    main()
