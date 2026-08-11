@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\.face-matching-cuda-ready" (
  echo First run: installing dependencies and models...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Backend cuda
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
set "FACE_MATCHING_GPU_BACKEND=cuda"
".venv\Scripts\python.exe" -m face_matching.app
if errorlevel 1 pause
