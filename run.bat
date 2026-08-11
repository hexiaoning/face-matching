@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The application is not installed yet. Starting install.bat...
    call install.bat
    if errorlevel 1 exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m face_matching
if errorlevel 1 (
    echo.
    echo Face Matching exited with an error. See the message above or the error dialog.
    pause
)
