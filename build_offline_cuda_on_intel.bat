@echo off
setlocal
cd /d "%~dp0"
echo Building the RTX CUDA package on a non-NVIDIA machine.
echo The frozen GPU self-test will run later on the RTX target via GPU诊断.bat.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_offline_bundle.ps1" -Backend cuda -SkipGpuSelfTest %*
if errorlevel 1 (
  echo.
  echo CUDA cross-machine bundle build failed. See the message above.
  pause
  exit /b 1
)
echo.
echo CUDA cross-machine bundle build completed.
pause
