@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Face Matching - Install

echo [1/5] Checking NVIDIA driver...
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: NVIDIA driver was not found. Install the latest NVIDIA driver first.
    echo CUDA Toolkit is NOT required; the Python packages include the CUDA runtime.
    pause
    exit /b 3
)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if errorlevel 1 (
    echo ERROR: The NVIDIA driver cannot access a GPU.
    pause
    exit /b 3
)

echo [2/5] Checking Python 3.12...
py -3.12 -V >nul 2>nul
if errorlevel 1 goto install_python
goto create_venv

:install_python
where winget >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.12 and winget are both unavailable.
    echo Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/ and run this file again.
    pause
    exit /b 2
)
echo Python 3.12 is missing. Installing it with winget...
winget install --exact --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
if errorlevel 1 (
    echo ERROR: Python installation failed.
    pause
    exit /b 2
)
set "PYTHON_DIRECT=%LocalAppData%\Programs\Python\Python312\python.exe"
if not exist "%PYTHON_DIRECT%" (
    echo ERROR: Python was installed but could not be located. Open a new terminal and run install.bat again.
    pause
    exit /b 2
)
"%PYTHON_DIRECT%" -m venv .venv
goto install_packages

:create_venv
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create the Python virtual environment.
    pause
    exit /b 2
)

:install_packages
echo [3/5] Updating the package installer...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto install_failed

echo [4/5] Installing GUI and bundled CUDA/cuDNN runtime packages...
python -m pip uninstall -y onnxruntime onnxruntime-directml >nul 2>nul
python -m pip install -e ".[gpu]"
if errorlevel 1 goto install_failed

echo [5/5] Verifying the mandatory CUDA execution provider...
python -m face_matching --check-gpu
if errorlevel 1 (
    echo.
    echo ERROR: Installation completed, but CUDA verification failed.
    echo Update the NVIDIA driver, then run install.bat again.
    pause
    exit /b 3
)

echo.
echo Installation completed. Double-click run.bat to start.
echo The face model will be downloaded and SHA-256 verified on first launch.
pause
exit /b 0

:install_failed
echo.
echo ERROR: Dependency installation failed. Review the messages above.
pause
exit /b 2

