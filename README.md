# A8mini Target Tracker v2.0

基于 SIYI A8mini 云台相机的目标图像学习识别系统。通过少量目标模板照片，在视频流中实时识别特定目标，无需训练神经网络。

## v2.0 变化 (MIPI CSI 集成)

- **MIPI CSI 直连采集**: 默认使用 Jetson MIPI CSI 接口，通过 GStreamer + nvarguscamerasrc 硬件加速取帧
- **延迟大幅降低**: MIPI ~50ms vs RTSP ~1000ms 首帧延迟，端到端延迟降低 20 倍
- **统一视频源接口**: `--source mipi|rtsp` 参数一键切换，RTSP 模式完全保留
- **实时 FPS 显示**: 画面左上角显示实际采集帧率
- **变焦拍摄工具也支持 MIPI**: `capture_zoom.py --source mipi` 直接 MIPI 取帧 + UDP 变焦
- **MIPICamera 模块**: 独立的 `mipi_camera.py`，支持曝光/增益参数调节，接口兼容 cv2.VideoCapture

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

## 性能对比 (MIPI vs RTSP)

| 指标 | MIPI CSI | RTSP (TCP) |
|------|----------|------------|
| 首帧延迟 | **~50ms** | ~1000ms |
| 连接建立 | ~20ms (GStreamer) | ~900ms (TCP握手) |
| 采集帧率 | 30fps (硬件保证) | ~25fps (网络依赖) |
| 端到端延迟 | **~80ms** | ~1100ms |
| CPU 占用 | 低 (NVMM硬件加速) | 较高 (FFmpeg解码) |

### 启动全链路耗时 (MIPI 模式)

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 模板加载 + 特征预计算 | ~87ms | 5个目标模板 |
| MIPI CSI 连接建立 | ~20ms | GStreamer pipeline |
| 收到第一帧 | **~120ms** | 从启动算起 |
| 首次识别到目标 | **~150ms** | 从启动算起 (快速模式) |
| 单帧识别 (快速模式) | **29ms** | ~34fps |
| 单帧识别 (全量模式) | ~1000ms | ~1fps |

## 使用方法

### 1. 采集目标照片

```bash
# MIPI 模式 (默认)
python capture_zoom.py

# RTSP 模式
python capture_zoom.py --source rtsp
```

- `+/-` 调整变焦 (1x/2x/3x/4x/6x)，变焦始终通过 UDP 控制
- `空格` 拍照保存到 `captures/`
- `q` 退出

### 2. 标注目标区域

```bash
python annotate.py
```

- 鼠标框选目标区域
- 自动裁剪保存到 `targets/`，生成 `target_info.json`

### 3. 实时识别

```bash
# MIPI 快速模式 (推荐)
python recognize.py --mipi --fast

# MIPI 全量模式
python recognize.py --mipi

# RTSP 快速模式
python recognize.py --rtsp rtsp://192.168.144.25:8554/main.264 --fast

# RTSP 全量模式
python recognize.py --rtsp rtsp://192.168.144.25:8554/main.264

# 带录像保存
python recognize.py --fast --save

# 批量测试图片 (不需要视频源)
python recognize.py --batch

# 单张图片测试
python recognize.py --image captures/zoom_1x.jpg
```

运行时按键：
- `f` 切换快速/全量模式
- `p` 暂停/继续识别
- `q` 退出

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mipi` | 关闭 | 使用 MIPI CSI 摄像头 (低延迟) |
| `--sensor-id N` | `0` | MIPI 摄像头编号 |
| `--width N` | `1280` | MIPI 采集宽度 |
| `--height N` | `720` | MIPI 采集高度 |
| `--fps-cap N` | `30` | MIPI 采集帧率 |
| `--fast` | 关闭 | 快速识别模式 (ORB+颜色, ~29ms) |
| `--save` | 关闭 | 录像保存到 `recordings/` |
| `--rtsp URL` | `rtsp://192.168.144.25:8554/main.264` | RTSP 视频流地址 |
| `--batch` | - | 批量测试 `captures/` 下所有图片 |
| `--image PATH` | - | 单张图片测试 |

## 硬件配置

- **计算平台：** Jetson Orin Nano (8GB)
- **相机：** SIYI A8mini 云台相机
- **视频采集 (MIPI)：** CSI 排线直连，nvarguscamerasrc，1280x720@30fps
- **视频采集 (RTSP)：** `rtsp://192.168.144.25:8554/main.264` (1280x720, HEVC, TCP)
- **云台控制：** UDP `192.168.144.25:37260` (SIYI 私有协议)

## 文件结构

```
a8mini_tracker/
├── recognize.py       # 核心识别引擎 (多方法 + MIPI/RTSP实时 + 录像)
├── capture_zoom.py    # MIPI/RTSP 多变焦抓图工具
├── mipi_camera.py     # MIPI CSI 采集模块 (GStreamer + nvarguscamerasrc)
├── annotate.py        # 目标区域标注/裁剪工具
├── siyi_sdk.py        # A8mini 云台协议 (变焦控制, UDP)
├── captures/          # 采集的场景图 (1x~4x 变焦)
├── targets/           # 目标模板 (裁剪图 + target_info.json)
└── recordings/        # 识别录像输出 (MP4, git忽略)
```

## 依赖

```
Python 3.8+
opencv-python (含 contrib, 需要 SIFT; Jetson 需 GStreamer 支持)
numpy
```

## 为什么不用 YOLO

本系统识别的是**特定个体目标**（拿到照片后找这个具体的东西），而非物体类别。YOLO 需要大量标注数据训练且只能识别类别，无法满足"临时换目标、秒切"的需求。传统视觉方案在本场景中更合适：换目标只需几张照片，无需训练，29ms 即可完成识别。

## Jetson 环境要求

- JetPack 5.x 或更高版本
- OpenCV 编译时开启 GStreamer 支持 (`-D WITH_GSTREAMER=ON`)
- `nvarguscamerasrc` 可用 (检查: `gst-inspect-1.0 nvarguscamerasrc`)
- CSI 排线正确连接 (建议先用 `nvgstcapture-1.0` 测试)

## 为什么用 MIPI 替代 RTSP

A8mini 原生通过以太网输出 RTSP 视频流，延迟约 1 秒。在 Jetson Orin Nano 上通过 CSI 排线直连 MIPI 摄像头，利用 nvarguscamerasrc 硬件 ISP 处理，端到端延迟降至 ~50ms，完全满足实时跟踪需求。RTSP 模式作为备用保留，方便远程调试。
