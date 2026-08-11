@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-directml\.face-matching-directml-ready" (
  echo First run: installing Intel DirectML dependencies and models...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Backend directml
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
set "FACE_MATCHING_GPU_BACKEND=directml"
".venv-directml\Scripts\python.exe" -m face_matching.app
if errorlevel 1 pause
