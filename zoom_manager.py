"""
Zoom管理器 — 锁定目标后自动放大，使目标占画面30-40%
目标持续居中0.6秒后允许zoom放大
远距离(目标<10%)允许跳2级加速放大
"""

import time
from config import (
    ZOOM_TABLE, ZOOM_PULSE_ON, ZOOM_PULSE_OFF, ZOOM_CHANGE_COOLDOWN,
    ZOOM_TARGET_RATIO, ZOOM_MAX,
)

# 居中稳定时间
CENTERED_STABLE_TIME = 0.4


class ZoomManager:
    def __init__(self, gimbal):
        self.gimbal = gimbal
        self.current_zoom = 1
        self.target_zoom = 1
        self.last_change_time = 0
        self.zooming = False
        self.zoom_direction = 0
        self.zoom_start = 0
        self.enabled = True
        # 居中稳定计时
        self.centered_since = 0         # 开始居中的时间（0=未居中）

    def update(self, box_ratio, center_error=1.0):
        """
        根据目标大小更新zoom，每帧调用一次
        center_error: 目标偏离中心的程度(0=正中, 1=边缘)
        """
        if not self.enabled or not self.gimbal:
            return

        now = time.time()

        # 更新居中稳定计时
        if center_error < 0.15:
            if self.centered_since == 0:
                self.centered_since = now
        else:
            self.centered_since = 0  # 偏离中心，重置计时

        centered_duration = now - self.centered_since if self.centered_since > 0 else 0

        # 简化zoom决策：目标<30%且还没到最大zoom → 想要更大zoom
        if (box_ratio < ZOOM_TARGET_RATIO and
                self.current_zoom < ZOOM_MAX and
                self.current_zoom >= self.target_zoom):  # 没有正在执行的zoom
            if (centered_duration >= CENTERED_STABLE_TIME and
                    now - self.last_change_time > ZOOM_CHANGE_COOLDOWN):
                # 远距离(目标很小)允许跳2级，近距离+1级
                if box_ratio < 0.10:
                    step = 2
                else:
                    step = 1
                next_zoom = min(self.current_zoom + step, ZOOM_MAX)
                self.target_zoom = next_zoom
                self.centered_since = 0  # 重置，zoom后需要重新稳定
                print(f'[Zoom] 居中{centered_duration:.1f}s size:{box_ratio:.1%} → zoom:{self.current_zoom}→{self.target_zoom}')

        # 执行zoom-in脉冲（每级一个脉冲）
        if self.current_zoom < self.target_zoom:
            if not self.zooming:
                self.gimbal.zoom_in()
                self.zooming = True
                self.zoom_direction = 1
                self.zoom_start = now
            elif now - self.zoom_start > ZOOM_PULSE_ON:
                self.gimbal.zoom_stop()
                self.zooming = False
                self.current_zoom += 1
                self.last_change_time = now
                # 如果还没到目标，短暂停顿后继续下一个脉冲
                if self.current_zoom < self.target_zoom:
                    time.sleep(0.1)  # 极短停顿

        # 已到达目标zoom，停止
        elif self.zooming:
            self.gimbal.zoom_stop()
            self.zooming = False
            self.zoom_direction = 0

    def set_search_zoom(self, elapsed_time, schedule):
        """搜索阶段根据时间表设置zoom"""
        if not self.gimbal:
            return

        target = 1
        for time_limit, zoom_level in schedule:
            if elapsed_time < time_limit:
                target = zoom_level
                break

        if target != self.current_zoom:
            if target < self.current_zoom:
                self.gimbal.zoom_out()
                time.sleep(ZOOM_PULSE_ON)
                self.gimbal.zoom_stop()
            else:
                self.gimbal.zoom_in()
                time.sleep(ZOOM_PULSE_ON)
                self.gimbal.zoom_stop()
            self.current_zoom = target
            print(f'[Zoom] 搜索zoom切换到 {target}x')

    def zoom_to_max(self):
        self.target_zoom = 6
        self.current_zoom = 6
        if self.gimbal:
            self.gimbal.zoom_in()
            time.sleep(3.0)
            self.gimbal.zoom_stop()
        print('[Zoom] 已zoom到最大(6x)')

    def zoom_to_min(self):
        self.target_zoom = 1
        if self.gimbal:
            self.gimbal.zoom_out()
            time.sleep(3.0)
            self.gimbal.zoom_stop()
        self.current_zoom = 1
        print('[Zoom] 已zoom到最小(1x)')

    def stop(self):
        if self.gimbal and self.zooming:
            self.gimbal.zoom_stop()
            self.zooming = False
            self.zoom_direction = 0
