$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$launcher = Join-Path $PSScriptRoot ".venv\Scripts\face-matching.exe"
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "尚未安装。请先双击 install.bat。"
}
& $launcher
