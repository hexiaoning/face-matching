@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Backend directml
if errorlevel 1 (
  echo.
  echo Intel DirectML installation failed. See the message above.
  pause
  exit /b 1
)
echo.
echo Intel DirectML installation completed.
pause
