@echo off
rem FaceMatch 一键安装（Windows 10/11 x64）
setlocal
cd /d %~dp0

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 Python。请先安装 Python 3.10+ ^(https://www.python.org/downloads/^)，
    echo        安装时勾选 "Add python.exe to PATH"。
    pause
    exit /b 1
)

echo [1/3] 创建虚拟环境 .venv ...
python -m venv .venv
if errorlevel 1 goto :fail

echo [2/3] 安装依赖（含 CUDA 运行时库，约 1.5GB，请耐心等待）...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [3/3] 完成。首次运行 run.bat 时会引导下载人脸识别模型。
echo.
echo 注意：本软件必须使用 NVIDIA GPU。请确认已安装最新的 NVIDIA 显卡驱动
echo      （nvidia-smi 能正常显示显卡信息）。
pause
exit /b 0

:fail
echo [错误] 安装失败，请检查网络后重试。
pause
exit /b 1
