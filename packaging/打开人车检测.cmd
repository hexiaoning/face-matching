@echo off
setlocal
cd /d "%~dp0"
start "" /wait "%~dp0FaceMatching.exe" --video-detection
