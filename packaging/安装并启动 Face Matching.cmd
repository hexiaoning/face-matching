@echo off
setlocal
cd /d "%~dp0"
title Face Matching - Install and Start

echo Checking the commercial AuraFace model on the real GPU...
set "REPORT=%TEMP%\face-matching-auraface-%RANDOM%-%RANDOM%.json"
start "" /wait "%~dp0FaceMatching.exe" --diagnose --profile auraface --report "%REPORT%"
set "CHECK_RESULT=%ERRORLEVEL%"
if exist "%REPORT%" type "%REPORT%"
if exist "%REPORT%" del /q "%REPORT%"
if not "%CHECK_RESULT%"=="0" (
  echo.
  echo AuraFace GPU check failed. Send the output above to support.
  pause
  exit /b %CHECK_RESULT%
)

echo.
echo Checking the high-accuracy LVFace-B model on the real GPU...
set "REPORT=%TEMP%\face-matching-lvface-%RANDOM%-%RANDOM%.json"
start "" /wait "%~dp0FaceMatching.exe" --diagnose --profile lvface-b --report "%REPORT%"
set "CHECK_RESULT=%ERRORLEVEL%"
if exist "%REPORT%" type "%REPORT%"
if exist "%REPORT%" del /q "%REPORT%"
if not "%CHECK_RESULT%"=="0" (
  echo.
  echo LVFace-B GPU check failed. Send the output above to support.
  pause
  exit /b %CHECK_RESULT%
)

echo.
echo Check passed. Starting Face Matching...
start "" "%~dp0FaceMatching.exe"
exit /b 0
