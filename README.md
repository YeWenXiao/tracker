# A8mini Target Tracker

基于 SIYI A8mini 云台相机的目标图像学习识别追踪系统。通过少量目标模板照片，实时识别并追踪特定目标，无需训练神经网络。

## 版本

| 版本 | 分支 | 核心能力 | Commits | 状态 |
|------|------|---------|---------|------|
| v1.0 | main | 基线识别引擎（5种方法） | 1 | ✅ 稳定 |
| v1.5 | [v1.5-hotswap](../../tree/v1.5-hotswap) | 不停流热更换识别目标 | 25 | ✅ 9轮迭代 |
| v2.0 | [v2.0-mipi](../../tree/v2.0-mipi) | MIPI CSI 低延迟 + 追踪 | 30 | ✅ 9轮迭代 |

## 版本功能对比

| 功能 | v1.0 | v1.5 | v2.0 |
|------|------|------|------|
| 多方法识别引擎 (模板/ORB/SIFT/颜色/边缘) | ✅ | ✅ | ✅ |
| RTSP 视频流 | ✅ | ✅ | ✅ (fallback) |
| MIPI CSI 低延迟视频 | - | - | ✅ |
| CUDA 加速 (预处理+模板匹配) | - | - | ✅ |
| 卡尔曼+CSRT 多目标追踪 | - | - | ✅ |
| 云台自动跟踪 | - | - | ✅ |
| 热更换识别目标 (不停流) | - | ✅ | - |
| HTTP API + 管理页面 | - | ✅ | - |
| 目标分组管理 | - | ✅ | - |
| SSE 实时事件通知 | - | ✅ | - |
| 目标版本管理 + 回滚 | - | ✅ | - |
| 特征磁盘缓存 | - | ✅ | ✅ |
| ROI 区域识别 | - | - | ✅ |
| 多进程架构 | - | - | ✅ |
| Systemd 服务 | - | - | ✅ |
| API Token 认证 | - | ✅ | - |
| 识别报告导出 | - | ✅ | - |
| 单元测试 | - | - | ✅ |
| 性能 Benchmark | - | - | ✅ |

## 技术方案

采用多方法传统视觉识别引擎，5种方法互补：

| # | 方法 | 全量耗时 | 说明 |
|---|------|---------|------|
| 1 | 多尺度模板匹配 | ~570ms | 最可靠，得分>0.997 |
| 2 | ORB 特征匹配 | ~8ms | 2000特征点 + 单应性变换 |
| 3 | SIFT 特征匹配 | ~103ms | FLANN加速，KD-tree索引 |
| 4 | HSV 颜色反投影 | ~9ms | 形态学滤波 + 轮廓分析 |
| 5 | 边缘模板匹配 | ~335ms | Canny边缘 + 模板匹配 |

后处理：NMS (IoU=0.3) → 颜色直方图验证 → 返回 Top5 候选

### 快速模式 (--fast)

- 场景缩小到 1/3 分辨率 + 3个尺度的模板匹配
- 高置信度 (>0.8) 提前退出，跳过 ORB 和颜色反投影
- 跳过 SIFT 和边缘匹配
- **实测识别耗时：29ms/帧**

## 性能对比

| 指标 | v1.0 (RTSP) | v2.0 (MIPI) | v2.0 (MIPI+CUDA) |
|------|------------|-------------|-------------------|
| 模板加载 | 87ms | 87ms (首次) / 5ms (缓存) | 同左 |
| 首帧延迟 | ~1000ms | ~50ms | ~50ms |
| 单帧识别 (快速) | 29ms | 29ms | <10ms |
| 单帧识别 (全量) | ~1000ms | ~1000ms | ~300ms |
| 追踪模式帧率 | - | - | ~60fps |

## 快速开始

### 安装

```bash
git clone https://github.com/YeWenXiao/tracker.git
cd tracker
pip install -r requirements.txt

# v2.0: 检查 Jetson 环境
python check_env.py
```

### 使用 v1.0 基线

```bash
# 1. 采集目标照片
python capture_zoom.py

# 2. 标注目标区域
python annotate.py

# 3. 实时识别
python recognize.py --fast
```

### 使用 v1.5 (热更换目标)

```bash
git checkout v1.5-hotswap

# 实时识别 + HTTP API
python recognize.py --fast --api

# 浏览器打开管理页面
# http://localhost:5000

# 运行时按 r 重新加载目标，按 g 切换分组
```

### 使用 v2.0 (MIPI + 追踪)

```bash
git checkout v2.0-mipi

# MIPI 快速模式 + 云台自动跟踪
python recognize.py --fast --auto-track

# RTSP 模式 (非 Jetson 平台)
python recognize.py --fast --rtsp

# 多进程模式
python recognize.py --fast --multiprocess

# 部署为系统服务
sudo bash install.sh
```

## 硬件配置

- **相机：** SIYI A8mini 云台相机
- **计算平台：** Jetson Orin Nano (v2.0) / 任意 Linux (v1.0/v1.5)
- **视频流：** RTSP `rtsp://192.168.144.25:8554/main.264` 或 MIPI CSI
- **云台控制：** UDP `192.168.144.25:37260` (SIYI 私有协议)

## 文件结构 (v1.0 基线)

```
tracker/
├── recognize.py       # 核心识别引擎
├── capture_zoom.py    # 多变焦抓图工具
├── annotate.py        # 目标区域标注工具
├── siyi_sdk.py        # A8mini 云台协议
├── captures/          # 采集的场景图
├── targets/           # 目标模板
├── docs/              # 开发日志和版本记录
│   ├── README.md
│   ├── v1.5_changelog.md
│   ├── v2.0_changelog.md
│   ├── decisions.md
│   └── improvements.md
└── recordings/        # 识别录像输出
```

v1.5 额外文件：`target_server.py`, `target_history.py`, `annotate_live.py`
v2.0 额外文件：`mipi_camera.py`, `tracker.py`, `pipeline.py`, `benchmark.py`, `logger.py`, `config.yaml`, `check_env.py`, `install.sh`, `tracker.service`

## 开发记录

详细的版本迭代记录见 [docs/](docs/) 目录：
- [总览](docs/README.md) - 项目概要和版本规划
- [v1.5 变更日志](docs/v1.5_changelog.md) - 9轮迭代详细记录
- [v2.0 变更日志](docs/v2.0_changelog.md) - 9轮迭代详细记录
- [决策记录](docs/decisions.md) - 每轮优化方向的决策依据
- [改进点跟踪](docs/improvements.md) - 所有优化项状态

## 为什么不用 YOLO

本系统识别的是**特定个体目标**（拿到照片后找这个具体的东西），而非物体类别。YOLO 需要大量标注数据训练且只能识别类别，无法满足"临时换目标、秒切"的需求。传统视觉方案在本场景中更合适：换目标只需几张照片，无需训练，29ms 即可完成识别。
