"""
用A8mini拍摄目标照片集 — 为V2.0特征匹配准备素材

用法:
  python capture_target.py                    # 默认保存到 target_photos/
  python capture_target.py --out my_target/   # 指定输出目录

操作:
  1. 把云台对准目标
  2. 按 1-5 切换zoom级别 (1x/2x/3x/4x/6x)
  3. 把目标放在画面中央，按空格拍照
  4. 每个zoom级别至少拍1张，建议拍2-3张
  5. 拍完按 q 退出，会提示你标注目标框

拍摄建议:
  - 6x zoom: 拍远距离的目标（模拟搜索阶段发现目标）
  - 3x-4x zoom: 拍中距离（模拟追踪阶段）
  - 1x-2x zoom: 拍近距离（模拟接近阶段）
  - 目标尽量居中，但不需要很精确
  - 每个焦距可以拍多张不同角度
"""

import cv2
import os
import sys
import time
import json
import argparse
import threading

# 添加父目录到path以导入gimbal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gimbal import SIYIGimbal
from config import CAMERA_IP

ZOOM_LEVELS = {
    '1': (1, '1x 近距离'),
    '2': (2, '2x'),
    '3': (3, '3x 中距离'),
    '4': (4, '4x'),
    '5': (6, '6x 远距离'),
}

# zoom级别到文件名前缀
ZOOM_PREFIX = {
    1: 'near',
    2: 'near',
    3: 'mid',
    4: 'mid',
    6: 'far',
}


def do_zoom(gimbal, target_zoom, current_zoom):
    """通过zoom_in/zoom_out脉冲切换zoom级别"""
    if target_zoom == current_zoom:
        return current_zoom

    # A8mini zoom是连续的，用脉冲逼近
    diff = target_zoom - current_zoom
    if diff > 0:
        # 需要放大
        pulse_time = abs(diff) * 0.8
        gimbal.zoom_in()
        time.sleep(pulse_time)
        gimbal.zoom_stop()
    else:
        # 需要缩小
        pulse_time = abs(diff) * 0.8
        gimbal.zoom_out()
        time.sleep(pulse_time)
        gimbal.zoom_stop()

    time.sleep(0.5)
    return target_zoom


def main():
    parser = argparse.ArgumentParser(description='用A8mini拍摄目标照片集')
    parser.add_argument('--camera_ip', default=CAMERA_IP)
    parser.add_argument('--rtsp', default=None, help='手动指定RTSP地址')
    parser.add_argument('--out', default='target_photos', help='输出目录')
    parser.add_argument('--no_gimbal', action='store_true', help='不连接云台(只看RTSP)')
    args = parser.parse_args()

    out_dir = args.out
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 连接云台
    gimbal = None
    if not args.no_gimbal:
        gimbal = SIYIGimbal(args.camera_ip)
        print(f'[云台] 已连接: {args.camera_ip}')

    # 连接RTSP (TCP模式)
    if args.rtsp:
        rtsp_url = args.rtsp
    else:
        rtsp_url = f'rtsp://{args.camera_ip}:8554/main.264'

    print(f'[RTSP] 连接: {rtsp_url}')
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # 等待第一帧
    ret = False
    for _ in range(50):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.1)

    if not ret:
        print('[RTSP] 连接失败! 请检查A8mini是否开机。')
        sys.exit(1)

    print(f'[RTSP] 连接成功!')

    # 后台线程持续读帧，避免主循环阻塞
    latest_frame = [None]
    frame_lock = threading.Lock()
    rtsp_running = True

    def rtsp_reader():
        while rtsp_running:
            ret, frame = cap.read()
            if ret:
                with frame_lock:
                    latest_frame[0] = frame
            else:
                time.sleep(0.01)

    reader_thread = threading.Thread(target=rtsp_reader, daemon=True)
    reader_thread.start()
    time.sleep(0.5)  # 等第一帧

    current_zoom = 1
    photos = []  # [(filename, zoom_level), ...]
    count = {'far': 0, 'mid': 0, 'near': 0}

    print()
    print('=' * 50)
    print('  A8mini 目标照片集拍摄工具')
    print('=' * 50)
    print()
    print('操作说明:')
    print('  WASD   云台方向控制 (W上 S下 A左 D右)')
    print('  松开自动停止')
    print('  1-5    切换zoom (1=1x, 2=2x, 3=3x, 4=4x, 5=6x)')
    print('  空格   拍照保存')
    print('  c      云台回中')
    print('  q      退出并进入标注')
    print()
    print('建议: 每个焦距拍2-3张，目标尽量居中')
    print()

    GIMBAL_SPEED = 40   # 云台移动速度
    PULSE_TIME = 0.15   # 每次按键移动的时长(秒)，越小步子越小

    cv2.namedWindow('Capture Target', cv2.WINDOW_NORMAL)

    while True:
        with frame_lock:
            frame = latest_frame[0].copy() if latest_frame[0] is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        vis = frame.copy()

        # 十字准星（帮助对准）
        cs = 30
        cv2.line(vis, (w//2 - cs, h//2), (w//2 + cs, h//2), (0, 0, 255), 2)
        cv2.line(vis, (w//2, h//2 - cs), (w//2, h//2 + cs), (0, 0, 255), 2)

        # 状态信息
        prefix = ZOOM_PREFIX.get(current_zoom, 'mid')
        info = f'Zoom:{current_zoom}x ({prefix}) | far={count["far"]} mid={count["mid"]} near={count["near"]}'
        cv2.putText(vis, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, 'WASD=move X=stop  1-5=zoom  SPACE=capture  C=center  Q=quit', (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Capture Target', vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            prefix = ZOOM_PREFIX.get(current_zoom, 'mid')
            idx = count[prefix] + 1
            fname = f'{prefix}_{idx:02d}.jpg'
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, frame)
            count[prefix] += 1
            photos.append((fname, current_zoom))
            total = sum(count.values())
            print(f'  [{total}] 已保存: {fname} (zoom={current_zoom}x)')
        elif key == ord('w') and gimbal:
            gimbal.set_speed(0, GIMBAL_SPEED)
            time.sleep(PULSE_TIME)
            gimbal.stop()
        elif key == ord('s') and gimbal:
            gimbal.set_speed(0, -GIMBAL_SPEED)
            time.sleep(PULSE_TIME)
            gimbal.stop()
        elif key == ord('a') and gimbal:
            gimbal.set_speed(-GIMBAL_SPEED, 0)
            time.sleep(PULSE_TIME)
            gimbal.stop()
        elif key == ord('d') and gimbal:
            gimbal.set_speed(GIMBAL_SPEED, 0)
            time.sleep(PULSE_TIME)
            gimbal.stop()
        elif key == ord('c'):
            if gimbal:
                gimbal.center()
                print('  [云台] 回中')
        elif key != 255 and chr(key) in ZOOM_LEVELS and gimbal:
            target, desc = ZOOM_LEVELS[chr(key)]
            print(f'  [Zoom] 切换到 {desc}...')
            current_zoom = do_zoom(gimbal, target, current_zoom)

    rtsp_running = False
    reader_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()

    total = sum(count.values())
    if total == 0:
        print('\n没有拍照片，退出。')
        if gimbal:
            gimbal.close()
        return

    print(f'\n拍摄完成! 共 {total} 张 (far={count["far"]} mid={count["mid"]} near={count["near"]})')

    # 进入标注模式
    print('\n' + '=' * 50)
    print('  现在标注目标位置')
    print('  在每张图上用鼠标拖框框住目标，按空格确认')
    print('  按 s 跳过（用全图作为模板）')
    print('=' * 50)

    crops_info = {}
    draw_state = {'drawing': False, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'done': False}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            draw_state['drawing'] = True
            draw_state['x1'] = x
            draw_state['y1'] = y
            draw_state['x2'] = x
            draw_state['y2'] = y
            draw_state['done'] = False
        elif event == cv2.EVENT_MOUSEMOVE and draw_state['drawing']:
            draw_state['x2'] = x
            draw_state['y2'] = y
        elif event == cv2.EVENT_LBUTTONUP:
            draw_state['drawing'] = False
            draw_state['x2'] = x
            draw_state['y2'] = y
            draw_state['done'] = True

    cv2.namedWindow('Label Target', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Label Target', mouse_callback)

    for i, (fname, zoom) in enumerate(photos):
        fpath = os.path.join(out_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        draw_state['done'] = False
        draw_state['drawing'] = False
        confirmed = False

        print(f'\n[{i+1}/{total}] {fname} (zoom={zoom}x) — 拖框框住目标，空格确认，s跳过')

        while True:
            vis = img.copy()
            h, w = vis.shape[:2]

            cv2.putText(vis, f'{fname} ({zoom}x) [{i+1}/{total}]', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, 'Drag to select target | SPACE=confirm | S=skip', (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if draw_state['drawing'] or draw_state['done']:
                x1 = min(draw_state['x1'], draw_state['x2'])
                y1 = min(draw_state['y1'], draw_state['y2'])
                x2 = max(draw_state['x1'], draw_state['x2'])
                y2 = max(draw_state['y1'], draw_state['y2'])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('Label Target', vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord(' ') and draw_state['done']:
                x1 = min(draw_state['x1'], draw_state['x2'])
                y1 = min(draw_state['y1'], draw_state['y2'])
                x2 = max(draw_state['x1'], draw_state['x2'])
                y2 = max(draw_state['y1'], draw_state['y2'])
                if x2 - x1 > 5 and y2 - y1 > 5:
                    crops_info[fname] = [x1, y1, x2, y2]
                    print(f'  已标注: [{x1},{y1},{x2},{y2}]')
                    confirmed = True
                    break
            elif key == ord('s'):
                print(f'  跳过（用全图）')
                break
            elif key == ord('q'):
                break

    cv2.destroyAllWindows()

    # 保存target_info.json
    info = {
        'description': f'A8mini拍摄的目标照片集，共{total}张',
        'capture_date': time.strftime('%Y-%m-%d %H:%M'),
        'crops': crops_info,
    }
    info_path = os.path.join(out_dir, 'target_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f'\n已保存 target_info.json ({len(crops_info)} 个标注)')
    print(f'照片集目录: {out_dir}')
    print(f'\n下一步: python main.py --photos {args.out}')

    if gimbal:
        gimbal.close()


if __name__ == '__main__':
    main()
