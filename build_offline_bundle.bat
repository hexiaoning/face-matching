@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_offline_bundle.ps1" %*
if errorlevel 1 (
  echo.
  echo Offline bundle build failed. See the message above.
  pause
  exit /b 1
)
echo.
echo Offline bundle build completed.
pause
