@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    call install_and_run.bat
    exit /b %ERRORLEVEL%
)
".venv\Scripts\python.exe" -m face_match
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" pause
exit /b %APP_EXIT%

