#!/bin/bash

# ── 关键修复：切换到脚本所在目录，保证相对路径正确 ──
cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "  睿聘智模 v3.1  AI 智能面试系统  Mac/Linux 启动脚本"
echo "  当前目录：$(pwd)"
echo "============================================================"

# ── 检查 .env 文件 ──
if [ ! -f ".env" ]; then
    echo ""
    echo "[警告] 未找到 .env 文件！"
    echo "请执行以下命令创建并填写 API_KEY："
    echo "  cp .env.example .env"
    echo "  nano .env    # 或用其他编辑器打开，填写 API_KEY=sk-xxxxxxxx"
    echo ""
    exit 1
fi

# ── 安装依赖 ──
echo "[1/2] 安装/检查依赖包..."
pip3 install -r requirements.txt -q
if [ $? -ne 0 ]; then
    echo "[错误] 依赖安装失败，请检查网络或手动运行：pip3 install -r requirements.txt"
    exit 1
fi
echo "[1/2] 依赖安装完成 ✓"

# ── 延迟打开浏览器 ──
echo "[2/2] 启动服务器..."
sleep 2 && (open http://localhost:5000 2>/dev/null || xdg-open http://localhost:5000 2>/dev/null) &

echo ""
echo "  本地访问：http://localhost:5000"
echo "  外部分享：ngrok http 5000"
echo "  按 Ctrl+C 停止"
echo "============================================================"

python3 app.py