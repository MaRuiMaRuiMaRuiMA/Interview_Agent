@echo off
chcp 65001 > nul
title 睿聘智模 v3.1
echo.
echo ============================================================
echo   睿聘智模 v3.1  AI 智能面试系统  Windows 启动脚本
echo ============================================================
echo.
python --version > nul 2>&1
if errorlevel 1 (echo [错误] 未找到 Python，请安装 Python 3.10+ & pause & exit /b 1)
echo [1/2] 安装依赖包...
pip install -r requirements.txt -q --no-warn-script-location
echo [2/2] 启动服务器...
echo.
echo   本地访问：http://localhost:5000
echo   外部分享：另开命令行窗口运行 "ngrok http 5000"
echo   按 Ctrl+C 停止服务
echo ============================================================
start "" "http://localhost:5000"
python app.py
pause