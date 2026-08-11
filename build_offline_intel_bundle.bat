@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_offline_bundle.ps1" -Backend directml %*
if errorlevel 1 (
  echo.
  echo Intel DirectML offline bundle build failed. See the message above.
  pause
  exit /b 1
)
echo.
echo Intel DirectML offline bundle build completed.
pause
