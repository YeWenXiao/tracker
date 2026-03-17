#!/bin/bash
# A8mini Tracker v2.0 安装脚本
set -e

echo "安装 A8mini Tracker v2.0..."

# 复制文件
sudo mkdir -p /opt/tracker
sudo cp *.py /opt/tracker/
sudo cp config.yaml /opt/tracker/
if [ -d targets ]; then
    sudo cp -r targets /opt/tracker/
fi

# 安装依赖
echo "安装 Python 依赖..."
pip3 install opencv-python numpy pyyaml

# 安装 systemd 服务
sudo cp tracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tracker

echo ""
echo "安装完成。"
echo "  启动服务: sudo systemctl start tracker"
echo "  查看状态: sudo systemctl status tracker"
echo "  查看日志: sudo journalctl -u tracker -f"
echo "  停止服务: sudo systemctl stop tracker"
