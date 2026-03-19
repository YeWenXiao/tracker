# A8mini Target Tracker

基于 SIYI A8mini 云台相机的目标图像学习识别追踪系统。通过少量目标模板照片，实时识别并追踪特定目标，无需训练神经网络。

## 版本

| 版本 | 分支 | 核心能力 | 状态 |
|------|------|---------|------|
| v1.0 | main | 基线识别引擎（5种方法） | ✅ 稳定 |
| v1.5 | [v1.5-hotswap](../../tree/v1.5-hotswap) | 不停流热更换识别目标 | ✅ 9轮迭代 |
| **v1.8** | **[v1.8-realtime-track](../../tree/v1.8-realtime-track)** | **三状态机实时追踪 + 自动变焦** | **✅ 实测通过** |
| v2.0 | [v2.0-mipi](../../tree/v2.0-mipi) | MIPI CSI 低延迟 + 追踪 | ✅ 9轮迭代 |

## v1.8 新增功能

v1.8 是面向实战场景的追踪版本：接收目标照片集 → 在 RTSP 视频流中搜索目标 → 云台自动居中锁定 → 自动变焦放大至目标占画面 30-40%。

### 三状态机架构

```
SEARCH ──找到目标──→ TRACK ──目标占>40%──→ TERMINAL
  ↑                    │                      │
  └──丢失超时(3s)──────┘                      │
  ↑                                           │
  └──────────目标缩小(<28%)───────────────────┘
```

- **SEARCH**: 1x 大视野，先静止检测 2 秒（目标通常已在画面中），找不到再扫描
- **TRACK**: PID 自适应居中 + 目标居中后自动 zoom 放大
- **TERMINAL**: 目标占画面 >40%，精确锁定

### 核心特性

| 特性 | 说明 |
|------|------|
| 颜色搜索优先 | HSV 颜色搜索为主搜索手段，不依赖 YOLO 检测自定义目标 |
| YOLO 辅助 | best.pt 模型辅助检测，提高召回率 |
| 多特征融合匹配 | 颜色直方图(0.45) + ORB纹理(0.40) + YOLO置信度(0.15) |
| 自适应 PID | 4档参数随目标大小切换，远距离激进/近距离保守 |
| 智能变焦 | 居中0.4s后放大，远距离跳2级加速，只放大不缩小 |
| 延迟扫描 | 前2秒静止检测，避免扫描把目标扫出画面 |
| RTSP TCP | 强制 TCP 传输，解决 UDP 丢帧问题 |
| 颜色验证 | 追踪中持续用颜色验证，避免 ORB 波动误判丢失 |
| 录像保存 | 带 OSD 信息的录像自动保存到桌面 |
| Web 监控 | http://0.0.0.0:8080 实时查看画面和控制 |

### 实测性能（2026-03-19）

| 阶段 | 时间 | 动作 |
|------|------|------|
| 启动 | 0s | 云台回正 + zoom 1x |
| 发现目标 | <1s | 颜色搜索立刻锁定（目标在画面中） |
| PID 居中 | 1-3s | 高增益 PID 快速拉到中心 |
| Zoom 1→3x | ~3s | 居中0.4s后跳2级 |
| Zoom 3→5x | ~6s | 再次居中后跳2级 |
| Zoom 5→6x | ~9s | +1级到最大 |
| **稳定锁定** | **~10s** | **目标占画面 36%，居中稳定** |

## 版本功能对比

| 功能 | v1.0 | v1.5 | v1.8 | v2.0 |
|------|------|------|------|------|
| 多方法识别引擎 | ✅ | ✅ | - | ✅ |
| HSV 颜色搜索 + ORB 匹配 | - | - | ✅ | - |
| RTSP 视频流 | ✅ | ✅ | ✅ | ✅ (fallback) |
| RTSP TCP 强制模式 | - | - | ✅ | - |
| 三状态机追踪 | - | - | ✅ | - |
| PID 自适应云台控制 | - | - | ✅ | ✅ |
| 智能自动变焦（居中后放大） | - | - | ✅ | - |
| YOLO 辅助检测 | - | - | ✅ | - |
| 颜色持续验证 | - | - | ✅ | - |
| Web 监控页面 | - | - | ✅ | - |
| 热更换识别目标 | - | ✅ | - | - |
| HTTP API + 管理页面 | - | ✅ | - | - |
| MIPI CSI 低延迟视频 | - | - | - | ✅ |
| CUDA 加速 | - | - | - | ✅ |
| 卡尔曼+CSRT 追踪 | - | - | - | ✅ |
| 特征磁盘缓存 | - | ✅ | - | ✅ |
| Systemd 服务 | - | - | - | ✅ |

## 快速开始

### 安装

```bash
git clone https://github.com/YeWenXiao/tracker.git
cd tracker
pip install -r requirements.txt
```

### 使用 v1.8（实时追踪）

```bash
git checkout v1.8-realtime-track

# 1. 准备目标照片集到 target_photos/（多焦距拍几张）
python capture_target.py

# 2. 启动追踪（全功能：云台+变焦+追踪+录像）
python main.py

# 3. 无云台模式（纯识别验证）
python main.py --no_gimbal --no_zoom

# 4. 用视频文件测试
python main.py --source video.mp4 --no_gimbal
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
python recognize.py --fast --api
# 浏览器打开 http://localhost:5000
```

## 硬件配置

- **相机：** SIYI A8mini 云台相机
- **计算平台：** Jetson Orin Nano / Windows / 任意 Linux
- **视频流：** RTSP `rtsp://192.168.144.25:8554/main.264`
- **云台控制：** UDP `192.168.144.25:37260` (SIYI 私有协议)

## 文件结构

### v1.8 (三状态机追踪)

```
tracker/
├── main.py              # 三状态机主循环 (SEARCH→TRACK→TERMINAL)
├── config.py            # 参数集中管理 (PID/Zoom/阈值)
├── gimbal.py            # SIYI A8mini 云台控制 (UDP协议)
├── pid.py               # PID 控制器 (4档自适应)
├── video.py             # RTSP 视频流读取 (TCP模式)
├── detector.py          # YOLO 检测器
├── feature_bank.py      # 目标特征库 (HSV+ORB预提取)
├── target_matcher.py    # 多特征融合匹配 (颜色+纹理+YOLO)
├── zoom_manager.py      # 智能变焦管理 (居中后放大)
├── web.py               # Web 监控服务
├── capture_target.py    # 目标照片采集工具
├── target_photos/       # 目标照片集
│   └── target_info.json
├── captures/            # 采集的场景图
├── targets/             # v1.0 目标模板
└── docs/                # 开发日志
```

### v1.8 关键参数 (config.py)

```python
# PID 四档 (kp, ki, kd, max_out) — 远距离激进居中，近距离保守防振荡
'far':      (30, 0.05, 4.0, 80)    # 目标 <5%  — 1x zoom，全力居中
'mid':      (25, 0.05, 5.0, 60)    # 目标 5%-20%
'near':     (12, 0.05, 5.0, 35)    # 目标 20%-50% — 高zoom保守
'terminal': (10, 0.05, 6.0, 30)    # 目标 >50%

# Zoom 策略
# 目标 < 30% → 继续放大（最大6x）
# 居中0.4s后启动，远距离(<10%)跳2级，冷却0.5s
ZOOM_TARGET_RATIO = 0.30
ZOOM_MAX = 6

# 匹配阈值
MATCH_THRESHOLD = 0.70       # YOLO+特征匹配
COLOR_SEARCH_THRESHOLD = 0.50 # 颜色搜索
VERIFY_THRESHOLD = 0.40       # 追踪验证(纯颜色)
```

## 2026-03-19 实测迭代记录

经过 5 轮实测迭代，从初始版本优化到当前稳定版：

1. **RTSP TCP 模式** — 强制 TCP 传输 + FFMPEG 后端，解决 Windows UDP 丢帧
2. **特征验证改用纯颜色** — ORB 对实时裁剪不稳定，验证时只用颜色直方图
3. **HSV 范围只算一次** — 初始化时计算并缓存，避免重复计算
4. **先检测再扫描** — 搜索阶段先检测当前帧，找到就立刻追踪
5. **延迟扫描启动** — 前 2 秒静止检测，适配"目标已在画面中"的场景
6. **Zoom 居中前置条件** — 目标必须连续居中 0.4s 才放大，防止目标飞出画面
7. **Zoom 跳级加速** — 远距离(目标<10%)允许跳 2 级，10s 内达到最大 zoom
8. **PID 远近分离** — 远距离 kp=30 全力居中(3s内)，高zoom时 kp=12 防振荡
9. **启动强制 zoom 1x** — 确保初始状态是大视野

## 开发记录

详细的版本迭代记录见 [docs/](docs/) 目录：
- [总览](docs/README.md) - 项目概要和版本规划
- [v1.5 变更日志](docs/v1.5_changelog.md) - 9轮迭代详细记录
- [v2.0 变更日志](docs/v2.0_changelog.md) - 9轮迭代详细记录
- [决策记录](docs/decisions.md) - 每轮优化方向的决策依据
- [改进点跟踪](docs/improvements.md) - 所有优化项状态

## 为什么不用 YOLO 作为主搜索

本系统识别的是**特定个体目标**（拿到照片后找这个具体的东西），而非物体类别。YOLO 只能识别训练过的类别，对自定义目标无效。v1.8 采用 HSV 颜色搜索为主 + YOLO 辅助的方案：
- 颜色搜索不需要训练，从照片集自动计算 HSV 范围
- YOLO 提高召回率，但不作为主要依赖
- 换目标只需几张照片，无需训练
