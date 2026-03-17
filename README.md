# A8mini Target Tracker v1.5

基于 SIYI A8mini 云台相机的目标图像学习识别系统。通过少量目标模板照片，在 RTSP 视频流中实时识别特定目标，无需训练神经网络。

## v1.5 新功能: 不停流热更换识别目标

**核心改进：** 识别引擎运行过程中，无需停止视频流即可更换识别目标。

### 热加载机制

- **手动触发：** 识别界面按 `r` 键立即重载 targets/ 目录
- **自动监控：** 后台线程每 2 秒检查 `targets/target_info.json` 的修改时间，变化时自动重载
- **HTTP API：** 通过 REST API 远程管理目标模板
- **线程安全：** 使用 Python GIL 原子赋值，识别线程不会因重载而崩溃

### HTTP API (target_server.py)

独立启动：
```bash
python target_server.py
```

或作为 recognize.py 的子线程（需代码集成）。端口 5000。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/targets` | 获取当前目标列表 |
| POST | `/api/targets/upload` | 上传新目标图片 (multipart, field="image") |
| POST | `/api/targets/reload` | 触发重新加载 |
| DELETE | `/api/targets/<name>` | 删除某个目标 |

示例：
```bash
# 查看目标列表
curl http://localhost:5000/api/targets

# 上传新目标
curl -X POST -F "image=@new_target.jpg" http://localhost:5000/api/targets/upload

# 手动触发重载
curl -X POST http://localhost:5000/api/targets/reload

# 删除目标
curl -X DELETE http://localhost:5000/api/targets/target_003.jpg
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

## 使用方法

### 1. 采集目标照片
```bash
python capture_zoom.py
# +/- 调整变焦, 空格拍照, q 退出
```

### 2. 标注目标区域
```bash
python annotate.py        # 离线 (从已有图片)
python annotate_live.py   # 实时 (从 RTSP 流)
```

### 3. 实时识别
```bash
python recognize.py --fast         # 快速模式
python recognize.py --fast --save  # 快速模式 + 录像
python recognize.py                # 全量模式
python recognize.py --batch        # 批量测试
```
运行时按键: `f`=切换模式 `p`=暂停 `r`=重载目标 `q`=退出

### 4. 目标管理服务器
```bash
python target_server.py  # 启动 HTTP API (端口 5000)
```

## 文件结构

```
a8mini_tracker/
|-- recognize.py       # 核心识别引擎 (多方法 + 实时 + 热加载)
|-- target_server.py   # 目标管理 HTTP API (Flask)
|-- annotate_live.py   # RTSP 实时视频流标注工具
|-- capture_zoom.py    # RTSP 多变焦抓图工具
|-- annotate.py        # 目标区域标注工具 (离线)
|-- siyi_sdk.py        # A8mini 云台协议 (变焦控制)
|-- captures/          # 采集的场景图
|-- targets/           # 目标模板 + target_info.json
|-- recordings/        # 识别录像输出
```

## 依赖

```
Python 3.8+
opencv-python (含 contrib, 需要 SIFT)
numpy
flask  # target_server.py
```
