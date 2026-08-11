$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "Face Matching is not installed. Run install.cmd first."
}

Set-Location $ProjectRoot
& $VenvPython -m face_matching
exit $LASTEXITCODE
