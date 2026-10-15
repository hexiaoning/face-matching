@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Face Matching v2.6.3 - Install and Start

echo ============================================================
echo Face Matching Offline Bundle v2.6.3
echo Build marker: DUAL-REVIEW-BUTTONS-20261015
echo ============================================================
echo.

echo Checking the bundled Qt GUI runtime...
set "REPORT=%TEMP%\face-matching-gui-%RANDOM%-%RANDOM%.json"
start "" /wait "%~dp0FaceMatching.exe" --check-gui --report "%REPORT%"
set "CHECK_RESULT=%ERRORLEVEL%"
if exist "%REPORT%" type "%REPORT%"
if not "%CHECK_RESULT%"=="0" (
  echo.
  echo Qt GUI runtime check failed. Send the report above to support.
  echo Diagnostic report saved to: "%REPORT%"
  pause
  exit /b %CHECK_RESULT%
)
if exist "%REPORT%" del /q "%REPORT%"
echo.

echo Checking the commercial AuraFace model on the real GPU...
set "REPORT=%TEMP%\face-matching-auraface-%RANDOM%-%RANDOM%.json"
start "" /wait "%~dp0FaceMatching.exe" --diagnose --profile auraface --report "%REPORT%"
set "CHECK_RESULT=%ERRORLEVEL%"
if exist "%REPORT%" type "%REPORT%"
if not "%CHECK_RESULT%"=="0" (
  echo.
  echo AuraFace GPU check failed. Send the output above to support.
  echo Diagnostic report saved to: "%REPORT%"
  pause
  exit /b %CHECK_RESULT%
)
if exist "%REPORT%" del /q "%REPORT%"

echo.
echo Checking the high-accuracy LVFace-B model on the real GPU...
set "REPORT=%TEMP%\face-matching-lvface-%RANDOM%-%RANDOM%.json"
start "" /wait "%~dp0FaceMatching.exe" --diagnose --profile lvface-b --report "%REPORT%"
set "CHECK_RESULT=%ERRORLEVEL%"
if exist "%REPORT%" type "%REPORT%"
if not "%CHECK_RESULT%"=="0" (
  echo.
  echo LVFace-B GPU check failed. Send the output above to support.
  echo Diagnostic report saved to: "%REPORT%"
  pause
  exit /b %CHECK_RESULT%
)
if exist "%REPORT%" del /q "%REPORT%"

echo.
echo Check passed. Starting Face Matching...
start "" "%~dp0FaceMatching.exe"
exit /b 0
