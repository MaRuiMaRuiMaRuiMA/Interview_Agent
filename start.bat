@echo off
chcp 65001 > nul
title 睿聘智模 v3.1

:: ── 关键修复：切换到脚本所在目录，保证相对路径正确 ──
cd /d "%~dp0"

echo.
echo ============================================================
echo   睿聘智模 v3.1  AI 智能面试系统  Windows 启动脚本
echo   当前目录：%CD%
echo ============================================================
echo.

:: ── 检查 Python ──
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10 或更高版本
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

:: ── 显示 Python 版本 ──
echo [信息] 检测到 Python 版本：
python --version

:: ── 检查 .env 文件 ──
if not exist ".env" (
    echo.
    echo [警告] 未找到 .env 文件！
    echo 请复制 .env.example 并重命名为 .env，然后填写您的 API_KEY：
    echo   copy .env.example .env
    echo   记事本打开 .env，填写：API_KEY=sk-xxxxxxxx
    echo.
    pause
    exit /b 1
)

:: ── 安装依赖 ──
echo.
echo [1/2] 安装/检查依赖包（首次运行较慢，请耐心等待）...
pip install -r requirements.txt -q --no-warn-script-location
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络连接或手动运行：
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)
echo [1/2] 依赖安装完成 ✓

:: ── 启动服务 ──
echo [2/2] 启动服务器...
echo.
echo ============================================================
echo   本地访问：http://localhost:5000
echo   外部分享：另开命令行窗口运行 "ngrok http 5000"
echo   按 Ctrl+C 停止服务
echo ============================================================
echo.

:: 延迟2秒后自动打开浏览器
timeout /t 2 /nobreak > nul
start "" "http://localhost:5000"

python app.py
if errorlevel 1 (
    echo.
    echo [错误] 服务器启动失败，请检查上方错误信息
    pause
)