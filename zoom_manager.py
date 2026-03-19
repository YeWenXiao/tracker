"""
Zoom管理器 — 锁定目标后自动放大，使目标占画面30-40%
支持双向zoom：目标太小则放大，目标太大则缩小
"""

import time
from config import (
    ZOOM_TABLE, ZOOM_PULSE_ON, ZOOM_PULSE_OFF, ZOOM_CHANGE_COOLDOWN,
)


class ZoomManager:
    """
    追踪阶段根据目标大小自动调整zoom:
    - 目标太小 → zoom放大（让目标占更多画面）
    - 目标太大 → zoom缩小（防止丢失）
    通过脉冲式zoom实现平滑过渡。
    """

    def __init__(self, gimbal):
        self.gimbal = gimbal
        self.current_zoom = 1
        self.target_zoom = 1
        self.last_change_time = 0
        self.zooming = False
        self.zoom_direction = 0     # 1=放大中, -1=缩小中
        self.zoom_start = 0
        self.enabled = True

    def update(self, box_ratio, center_error=1.0):
        """
        根据目标大小更新zoom，每帧调用一次
        center_error: 目标偏离中心的程度(0=正中, 1=边缘)，只有<0.15才允许zoom放大
        """
        if not self.enabled or not self.gimbal:
            return

        now = time.time()

        # 从ZOOM_TABLE查目标zoom级别
        new_target = 1
        for max_ratio, zoom_level in ZOOM_TABLE:
            if box_ratio < max_ratio:
                new_target = zoom_level
                break

        # zoom级别变化且冷却期已过，且目标已居中
        if new_target > self.target_zoom:
            # 放大：必须目标居中才行
            if center_error < 0.15 and now - self.last_change_time > ZOOM_CHANGE_COOLDOWN:
                old = self.target_zoom
                self.target_zoom = new_target
                print(f'[Zoom] 目标已居中(err:{center_error:.2f}) size:{box_ratio:.1%} → 放大 zoom:{old}→{self.target_zoom}')

        # 需要zoom-in（放大）
        if self.current_zoom < self.target_zoom:
            if not self.zooming:
                self.gimbal.zoom_in()
                self.zooming = True
                self.zoom_direction = 1
                self.zoom_start = now
            elif now - self.zoom_start > ZOOM_PULSE_ON:
                self.gimbal.zoom_stop()
                self.zooming = False
                self.current_zoom = min(self.current_zoom + 1, self.target_zoom)
                self.last_change_time = now

        # 已到达或超过目标zoom，不缩小，保持当前zoom
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
