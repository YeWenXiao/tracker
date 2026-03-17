# A8mini Target Tracker v1.5

基于 SIYI A8mini 云台相机的目标图像学习识别系统。通过少量目标模板照片，在 RTSP 视频流中实时识别特定目标，无需训练神经网络。

## v1.5 新功能: 不停流热更换识别目标

**核心改进：** 识别引擎运行过程中，无需停止视频流即可更换识别目标。

### Round 5 新增

- **目标相似度去重：** 上传新目标时自动检测与已有目标的相似度（颜色直方图+ORB特征），防止重复添加。相似度>80%时发出警告。新增 `POST /api/targets/check-similarity` 接口支持仅检查不保存。
- **识别置信度自适应：** `AdaptiveThreshold` 根据最近200帧识别得分自动调整阈值（平均得分的70%，平滑更新），OSD 实时显示当前阈值。目标自定义的 `min_confidence` 优先级高于自适应值。
- **管理页面增强：** SSE 推送 `targets_changed` 事件实现多客户端自动同步；识别统计面板每5秒刷新；上传时显示相似度警告。
- **统计 API：** `GET /api/stats` 返回识别统计数据和自适应阈值。

### 热加载机制

- **增量加载：** 只计算新增模板的特征，保留未变化模板的缓存，避免全量重算
- **手动触发：** 识别界面按 `r` 键立即重载 targets/ 目录
- **自动监控：** 后台线程每 2 秒检查 `targets/target_info.json` 的修改时间，变化时自动重载
- **HTTP API：** 通过 REST API 远程管理目标模板（可内嵌到识别引擎或独立运行）
- **线程安全：** 使用 Python GIL 原子赋值，识别线程不会因重载而崩溃

### HTTP API

**方式一：内嵌到识别引擎（推荐）**
```bash
python recognize.py --fast --api                  # 默认端口 5000
python recognize.py --fast --api --api-port 8080  # 自定义端口
```

**方式二：独立运行**
```bash
python target_server.py
```

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 目标管理 HTML 页面（缩略图+上传+删除+权重调整） |
| GET | `/api/targets` | 获取当前目标列表 |
| POST | `/api/targets/upload` | 上传新目标图片 (含相似度去重检测, 可选 weight/min_confidence) |
| POST | `/api/targets/check-similarity` | 上传图片检查相似度（不保存） |
| POST | `/api/targets/reload` | 触发重新加载 |
| PUT | `/api/targets/<name>` | 更新目标权重和最低置信度 |
| DELETE | `/api/targets/<name>` | 删除某个目标 |
| GET | `/api/targets/<name>/preview` | 获取目标模板缩略图 |
| GET | `/api/targets/export` | 导出当前目标为 ZIP |
| POST | `/api/targets/import` | 从 ZIP 导入目标（替换当前目标） |
| GET | `/api/history` | 列出目标模板历史快照 |
| POST | `/api/history/rollback` | 回滚到指定快照 |
| GET | `/api/history/detections` | 最近N次识别结果 |
| GET | `/api/stats` | 识别统计信息 |
| GET | `/api/events` | SSE 实时事件流（识别结果+重载通知+targets_changed） |

示例：
```bash
# 查看目标列表
curl http://localhost:5000/api/targets

# 上传新目标（带权重和最低置信度）
curl -X POST -F "image=@new_target.jpg" -F "weight=1.5" -F "min_confidence=0.6" \
     http://localhost:5000/api/targets/upload

# 更新目标权重
curl -X PUT -H "Content-Type: application/json" \
     -d '{"weight": 2.0, "min_confidence": 0.5}' \
     http://localhost:5000/api/targets/target_000.jpg

# 手动触发重载
curl -X POST http://localhost:5000/api/targets/reload

# 删除目标
curl -X DELETE http://localhost:5000/api/targets/target_003.jpg

# 监听 SSE 实时事件
curl http://localhost:5000/api/events

# 浏览器打开管理页面
# http://localhost:5000/
```

### 目标置信度权重系统

每个目标模板可以独立设置权重和最低置信度阈值：

```json
{
  "source": "zoom_1x.jpg",
  "crop": "target_000.jpg",
  "bbox": [100, 200, 300, 400],
  "image_size": [1280, 720],
  "weight": 1.5,
  "min_confidence": 0.6
}
```

- **weight** (默认 1.0): 最终得分乘以此权重，用于提高/降低特定目标的优先级
- **min_confidence** (默认 0.45): 颜色验证阶段的最低置信度阈值

### WebSocket 实时通知 (SSE)

通过 Server-Sent Events 推送实时事件，无需额外依赖：

- **detection**: 每帧最佳识别结果 (score, method, bbox)
- **reload**: 目标模板重载通知 (added, removed, count)
- **heartbeat**: 30秒无事件时发送心跳保持连接

### Round 4: 版本管理 + 批量导入导出 + 识别统计

#### 目标模板版本管理 (target_history.py)

每次 reload 时自动保存当前目标集快照，支持查看历史和回滚：

```bash
# 列出历史快照
curl http://localhost:5000/api/history

# 回滚到指定快照
curl -X POST -H "Content-Type: application/json"      -d '{"snapshot": "20260317_143000_reload_+1_-0"}'      http://localhost:5000/api/history/rollback
```

#### 批量导入/导出

```bash
# 导出当前目标为 ZIP
curl -o targets.zip http://localhost:5000/api/targets/export

# 从 ZIP 导入目标（替换当前所有目标）
curl -X POST -F "file=@targets.zip" http://localhost:5000/api/targets/import
```

#### 识别统计

```bash
# 查看识别统计信息（平均耗时、检测率等）
curl http://localhost:5000/api/stats

# 查看最近N次识别结果
curl http://localhost:5000/api/history/detections?n=20
```

### 实时视频标注 (annotate_live.py)

在 RTSP 实时视频流上直接框选新目标：

```bash
# 基本用法
python annotate_live.py

# 指定 RTSP 地址
python annotate_live.py --rtsp rtsp://192.168.144.25:8554/main.264

# 通过 HTTP API 触发重载
python annotate_live.py --api http://localhost:5000
```

操作：
- 鼠标拖框选区域
- `s` 保存裁剪到 targets/，自动更新 target_info.json
- `q` 退出

保存后自动触发 recognize.py 的热加载（通过文件监控或 HTTP API）。

## 技术方案

采用多方法传统视觉识别引擎，5种方法互补：

| # | 方法 | 全量耗时 | 说明 |
|---|------|---------|------|
| 1 | 多尺度模板匹配 | ~570ms | 最可靠，得分>0.997 |
| 2 | ORB 特征匹配 | ~8ms | 2000特征点 + 单应性变换 |
| 3 | SIFT 特征匹配 | ~103ms | FLANN加速，KD-tree索引 |
| 4 | HSV 颜色反投影 | ~9ms | 形态学滤波 + 轮廓分析 |
| 5 | 边缘模板匹配 | ~335ms | Canny边缘 + 模板匹配 |

后处理：NMS (IoU=0.3) -> 颜色直方图验证 -> 返回 Top5 候选

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
| 增量热加载 (新增1个模板) | ~17ms | vs 全量重算 87ms |

## 使用方法

### 1. 采集目标照片

```bash
python capture_zoom.py
```

- `+/-` 调整变焦 (1x/2x/3x/4x/6x)
- `空格` 拍照保存到 `captures/`
- `q` 退出

### 2. 标注目标区域

离线标注（从已有图片）：
```bash
python annotate.py
```

实时标注（从 RTSP 流）：
```bash
python annotate_live.py
```

- 鼠标框选目标区域
- 自动裁剪保存到 `targets/`，生成 `target_info.json`

### 3. 实时识别

```bash
# 快速模式 (~29ms/帧)
python recognize.py --fast

# 快速模式 + 录像保存
python recognize.py --fast --save

# 快速模式 + HTTP API 服务器
python recognize.py --fast --api

# 快速模式 + API + 自定义端口
python recognize.py --fast --api --api-port 8080

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
- `r` 手动重载目标模板
- `q` 退出

### 4. 目标管理服务器（可选）

```bash
python target_server.py
```

提供 HTTP API 远程管理目标模板，支持上传、删除、重载。
也可通过 `--api` 参数直接内嵌到 recognize.py 中。

## 硬件配置

- **相机：** SIYI A8mini 云台相机
- **视频流：** RTSP `rtsp://192.168.144.25:8554/main.264` (1280x720, HEVC)
- **云台控制：** UDP `192.168.144.25:37260` (SIYI 私有协议)
- **传输模式：** TCP (避免 UDP 丢包)

## 文件结构

```
a8mini_tracker/
├── recognize.py       # 核心识别引擎 (多方法 + 实时 + 热加载 + API集成)
├── target_history.py  # 目标模板版本管理 (快照/回滚)
├── target_server.py   # 目标管理 HTTP API 服务器 (Flask)
├── annotate_live.py   # RTSP 实时视频流目标标注工具
├── capture_zoom.py    # RTSP 多变焦抓图工具
├── annotate.py        # 目标区域标注/裁剪工具 (离线)
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
flask          # target_server.py 需要
```

## 为什么不用 YOLO

本系统识别的是**特定个体目标**（拿到照片后找这个具体的东西），而非物体类别。YOLO 需要大量标注数据训练且只能识别类别，无法满足"临时换目标、秒切"的需求。传统视觉方案在本场景中更合适：换目标只需几张照片，无需训练，29ms 即可完成识别。
