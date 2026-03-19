"""
PID 控制器 — 支持动态参数切换
"""

from config import PID_PROFILES, PID_DEADZONE


class PID:
    def __init__(self, kp=0.8, ki=0.01, kd=0.1, max_out=80):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral = 0.0
        self.prev_err = 0.0

    def compute(self, error, dt=0.033):
        if abs(error) < PID_DEADZONE:
            return 0.0
        self.integral += error * dt
        self.integral = max(-50, min(50, self.integral))
        deriv = (error - self.prev_err) / dt if dt > 0 else 0
        self.prev_err = error
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        return max(-self.max_out, min(self.max_out, out))

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0

    def set_profile(self, box_ratio):
        """根据目标大小自动切换PID参数"""
        if box_ratio < 0.05:
            profile = PID_PROFILES['far']
        elif box_ratio < 0.20:
            profile = PID_PROFILES['mid']
        elif box_ratio < 0.50:
            profile = PID_PROFILES['near']
        else:
            profile = PID_PROFILES['terminal']

        self.kp, self.ki, self.kd, self.max_out = profile
