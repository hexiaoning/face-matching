@echo off
rem FaceMatch 一键启动
setlocal
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
    echo 尚未安装依赖，正在运行 install.bat ...
    call install.bat
    if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe -m face_match.app
if errorlevel 1 pause
