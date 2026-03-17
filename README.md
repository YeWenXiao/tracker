# A8mini Target Tracker v1.0

基于 SIYI A8mini 云台相机的目标图像学习识别系统。通过少量目标模板照片，在 RTSP 视频流中实时识别特定目标，无需训练神经网络。

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

## 性能实测

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 模板加载 + 特征预计算 | 87ms | 5个目标模板 |
| RTSP 连接建立 | 900ms | TCP模式 |
| 收到第一帧 | 1000ms | 从启动算起 |
| 首次识别到目标 | **1098ms** | 从启动算起 |
| 单帧识别 (快速模式) | **29ms** | ~34fps |
| 单帧识别 (全量模式) | ~1000ms | ~1fps |

## 使用方法

### 1. 采集目标照片

```bash
python capture_zoom.py
```

- `+/-` 调整变焦 (1x/2x/3x/4x/6x)
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
# 快速模式 (~29ms/帧)
python recognize.py --fast

# 快速模式 + 录像保存
python recognize.py --fast --save

# 全量模式 (5种方法全跑)
python recognize.py

# 批量测试图片
python recognize.py --batch

# 单张图片测试
python recognize.py --image captures/zoom_1x.jpg
```

运行时按键：
- `f` 切换快速/全量模式
- `p` 暂停/继续识别
- `q` 退出

## 硬件配置

- **相机：** SIYI A8mini 云台相机
- **视频流：** RTSP `rtsp://192.168.144.25:8554/main.264` (1280×720, HEVC)
- **云台控制：** UDP `192.168.144.25:37260` (SIYI 私有协议)
- **传输模式：** TCP (避免 UDP 丢包)

## 文件结构

```
a8mini_tracker/
├── recognize.py       # 核心识别引擎 (多方法 + 实时视频流 + 录像)
├── capture_zoom.py    # RTSP 多变焦抓图工具
├── annotate.py        # 目标区域标注/裁剪工具
├── siyi_sdk.py        # A8mini 云台协议 (变焦控制)
├── captures/          # 采集的场景图 (1x~4x 变焦)
├── targets/           # 目标模板 (裁剪图 + target_info.json)
└── recordings/        # 识别录像输出 (MP4, git忽略)
```

## 依赖

```
Python 3.8+
opencv-python (含 contrib, 需要 SIFT)
numpy
```

## 为什么不用 YOLO

本系统识别的是**特定个体目标**（拿到照片后找这个具体的东西），而非物体类别。YOLO 需要大量标注数据训练且只能识别类别，无法满足"临时换目标、秒切"的需求。传统视觉方案在本场景中更合适：换目标只需几张照片，无需训练，29ms 即可完成识别。
