#!/bin/bash

echo "=========================================="
echo "SmartReach Accessibility Edition"
echo "Voice-Guided Object Grasping Assistant"
echo "=========================================="
echo ""

# 检查 Python 版本
python3 --version

# 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt

# 启动 FastAPI 服务
echo ""
echo "Starting FastAPI server with Uvicorn..."
echo "Open http://localhost:8000 in your browser (optional)"
echo "System will respond to voice commands through microphone"
echo ""

cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
