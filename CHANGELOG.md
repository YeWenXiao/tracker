# Changelog - v2.0-mipi

所有 v2.0 版本的变更记录。

## [v2.0.8] - 2026-03-17
### 新增
- 模板特征磁盘缓存（ORB/SIFT 预计算，启动加速 87ms→5ms）
- 多进程流水线架构（`--multiprocess`，采集/识别进程分离）
- Systemd 服务文件 `tracker.service` 和安装脚本 `install.sh`

## [v2.0.7] - 2026-03-17
### 新增
- 多目标追踪 `MultiTracker`（最多 5 个目标同时追踪）
- 追踪质量评估（连续性/稳定性/尺寸一致性评分）
- recognize.py 集成 MultiTracker + OSD 质量显示
- 视频录制增强：轨迹叠加 + 元数据 JSON 输出

## [v2.0.6] - 2026-03-17
### 新增
- 异常恢复机制（视频流重连 / CUDA OOM 回退 / 云台超时保护）
- 单元测试（`tests/` 目录，覆盖 TargetRecognizer + KalmanTracker）
- 命令行帮助分组显示 + 中文示例

## [v2.0.5] - 2026-03-17
### 新增
- 卡尔曼滤波追踪器（`tracker.py`，Kalman 预测 + CSRT 视觉追踪）
- 自适应识别频率（追踪稳定时降低识别频率，节省算力）
- 追踪器与 recognize.py 集成 + OSD 追踪状态显示
- 云台速率控制 `gimbal_rotate`（SIYI CMD 0x07）
- config.yaml 追踪配置项（dead_zone / speed_gain / max_lost_frames）

## [v2.0.4] - 2026-03-17
### 新增
- 运行状态 OSD 监控（CPU/GPU 温度 + 内存使用率）
- ROI 区域识别（鼠标拖框 / 配置文件预设 / 'd' 键切换）
### 修复
- 修复 `mipi_cam` 未定义变量问题

## [v2.0.3] - 2026-03-17
### 新增
- 日志系统（`logger.py`，用 Python logging 替代所有 print）
- 性能 benchmark 脚本（`benchmark.py`）
### 改进
- CUDA 加速模板匹配（`cv2.cuda.createTemplateMatching`）
- 识别线程帧丢弃策略（跳过 >500ms 的过时帧）

## [v2.0.2] - 2026-03-17
### 新增
- CUDA 加速预处理（cvtColor / resize GPU 加速）
- config.yaml 配置文件支持（YAML + fallback 纯文本解析）
- MIPI→RTSP 自动降级（MIPI 不可用时自动切换 RTSP）
### 改进
- 更新 README 说明 Round 2 改动

## [v2.0.1] - 2026-03-17
### 新增
- recognize.py 支持 MIPI CSI 视频源（GStreamer nvarguscamerasrc）
- capture_zoom.py 支持 MIPI 视频源
- `mipi_camera.py` 封装类（cv2 兼容 API：open/read/get_fps/release）
- MIPI CSI 设为默认视频源
### 修复
- MIPICamera API 兼容性修复（open/read/get_fps/release）
- capture_zoom.py MIPICamera API 调用修复

## [v2.0.0] - 2026-03-17
### 基础
- v1.0 图像学习识别系统（多尺度模板匹配 / ORB / SIFT / 颜色直方图 / 边缘轮廓）
