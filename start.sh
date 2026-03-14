#!/bin/bash
echo ""
echo "============================================================"
echo "  睿聘智模 v3.1  AI 智能面试系统  Mac/Linux 启动脚本"
echo "============================================================"
pip3 install -r requirements.txt -q
sleep 2 && (open http://localhost:5000 2>/dev/null || xdg-open http://localhost:5000 2>/dev/null) &
echo ""
echo "  本地访问：http://localhost:5000"
echo "  外部分享：ngrok http 5000"
echo "  按 Ctrl+C 停止"
echo "============================================================"
python3 app.py