@echo off
chcp 65001 >nul
setlocal
cd /d %~dp0

echo ==========================================
echo  视频人脸比对系统 - 一键安装并运行
echo ==========================================

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 Python。请先安装 Python 3.10+ 并勾选 "Add to PATH"。
    pause
    exit /b 1
)

if not exist .venv (
    echo [1/4] 创建虚拟环境...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

echo [2/4] 安装依赖（含 GPU 版 onnxruntime 与 CUDA 运行库）...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络后重试。
    pause
    exit /b 1
)

echo [3/4] 下载人脸识别模型（首次运行需要联网）...
python tools\download_models.py

echo [4/4] 启动程序（需要 NVIDIA GPU，未检测到 GPU 将报错退出）...
python -m app.main
if errorlevel 1 pause
