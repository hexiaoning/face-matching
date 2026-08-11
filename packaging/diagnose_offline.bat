@echo off
setlocal
cd /d "%~dp0"
"%~dp0FaceMatching.exe" --diagnose-file "%~dp0diagnostics.json"
set "RESULT=%ERRORLEVEL%"
type "%~dp0diagnostics.json"
echo.
if not "%RESULT%"=="0" (
  echo GPU/model diagnostics failed with exit code %RESULT%.
  pause
)
exit /b %RESULT%
