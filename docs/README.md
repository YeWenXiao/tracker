# Tracker 开发日志 - 主管记录

> 自动生成于 2026-03-17，记录 v1.5 / v2.0 / 后续版本的所有开发活动

## 项目概要

基于 SIYI A8mini 云台相机的目标图像学习识别系统。
- **仓库**: https://github.com/YeWenXiao/tracker
- **基线**: v1.0 - 多方法传统视觉识别引擎（模板匹配 + ORB + SIFT + 颜色反投影 + 边缘匹配）

## 版本规划

| 版本 | 分支 | 核心目标 | 状态 |
|------|------|---------|------|
| v1.0 | main | 基线识别系统 | 已完成 |
| v1.5 | v1.5-hotswap | 不停流热更换识别目标 | Round 9 ✅ (25 commits) |
| v2.0 | v2.0-mipi | MIPI CSI 替代 RTSP 降低延迟 | Round 9 ✅ (30 commits) |

## 文件索引

- [v1.5 开发日志](v1.5_changelog.md)
- [v2.0 开发日志](v2.0_changelog.md)
- [版本决策记录](decisions.md)
- [优化改进点跟踪](improvements.md)
