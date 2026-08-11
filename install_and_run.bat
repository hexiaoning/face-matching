@echo off
setlocal
cd /d "%~dp0"

set "PY_LAUNCHER="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>nul
    if not errorlevel 1 set "PY_LAUNCHER=py -3.12"
    if not defined PY_LAUNCHER (
        py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>nul
        if not errorlevel 1 set "PY_LAUNCHER=py -3.11"
    )
    if not defined PY_LAUNCHER (
        py -3.13 -c "import sys; assert sys.version_info[:2] == (3, 13)" >nul 2>nul
        if not errorlevel 1 set "PY_LAUNCHER=py -3.13"
    )
)

if not defined PY_LAUNCHER (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13)" >nul 2>nul
        if not errorlevel 1 set "PY_LAUNCHER=python"
    )
)

if not defined PY_LAUNCHER (
    where winget >nul 2>nul
    if not errorlevel 1 (
        echo Python was not found. Installing 64-bit Python 3.12 for the current user...
        winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
        if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY_LAUNCHER="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    )
)

if not defined PY_LAUNCHER (
    echo [ERROR] Python 3.11, 3.12 or 3.13 could not be installed automatically.
    echo Install 64-bit Python from https://www.python.org/downloads/windows/ and run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating isolated Python environment...
    %PY_LAUNCHER% -m venv .venv
    if errorlevel 1 goto :failed
)

echo Installing application and bundled CUDA/cuDNN runtime libraries...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip uninstall -y onnxruntime >nul 2>nul
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto :failed

echo Starting Face Match...
".venv\Scripts\python.exe" -m face_match
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" pause
exit /b %APP_EXIT%

:failed
echo.
echo [ERROR] Installation failed. Check the network connection and the messages above.
pause
exit /b 1
