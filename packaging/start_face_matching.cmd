@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Face Matching v2.8.0 - Offline Unified Analysis
if not exist "%~dp0FaceMatching.exe" (
  echo Please extract the complete offline ZIP before starting.
  pause
  exit /b 2
)
start "" "%~dp0FaceMatching.exe"
