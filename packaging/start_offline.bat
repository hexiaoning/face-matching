@echo off
setlocal
cd /d "%~dp0"
"%~dp0FaceMatching.exe"
if errorlevel 1 pause
