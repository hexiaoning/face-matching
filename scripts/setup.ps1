[CmdletBinding()]
param(
    [ValidateSet("cuda", "directml")]
    [string]$Backend = "cuda",
    [switch]$SkipGpuCheck
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvName = if ($Backend -eq "cuda") { ".venv" } else { ".venv-directml" }
$VenvPython = Join-Path $ProjectRoot "$VenvName\Scripts\python.exe"
$InstallMarker = Join-Path $ProjectRoot "$VenvName\.face-matching-$Backend-ready"
$env:FACE_MATCHING_GPU_BACKEND = $Backend

Set-Location $ProjectRoot

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): python $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $Created = $false
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) {
        foreach ($Version in @("3.12", "3.11", "3.13")) {
            & py "-$Version" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Creating Python $Version environment..."
                & py "-$Version" -m venv $VenvName
                if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $VenvPython)) {
                    $Created = $true
                    break
                }
            }
        }
    }
    if (-not $Created) {
        $Python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $Python) {
            throw "Python 3.11-3.13 was not found. Install 64-bit Python, then run the installer again."
        }
        & python -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,14)"
        if ($LASTEXITCODE -ne 0) {
            throw "Python must be a 64-bit Python 3.11-3.13 installation."
        }
        & python -m venv $VenvName
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
            throw "Failed to create the Python virtual environment."
        }
    }
}

& $VenvPython -c "import struct, sys; assert (3,11) <= sys.version_info[:2] < (3,14) and struct.calcsize('P') == 8"
if ($LASTEXITCODE -ne 0) {
    throw "The virtual environment must use 64-bit Python 3.11-3.13."
}
if (Test-Path -LiteralPath $InstallMarker) {
    Remove-Item -LiteralPath $InstallMarker -Force
}

Write-Host "Installing $Backend GPU runtime and desktop dependencies..."
Invoke-CheckedPython -m pip install --upgrade pip setuptools wheel
Invoke-CheckedPython -m pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu
Invoke-CheckedPython -Arguments @("-m", "pip", "install", "-e", ".[dev,$Backend]")
if (-not $SkipGpuCheck) {
    Invoke-CheckedPython -c "from face_matching.config import EngineConfig; from face_matching.gpu import assert_gpu_available; c=EngineConfig(); print('GPU provider:', assert_gpu_available(c.gpu_backend, c.prefer_tensorrt))"
}

Write-Host "Downloading research model weights (about 750 MB)..."
Write-Host "NOTICE: bundled download sources permit pretrained weights for non-commercial research only."
Invoke-CheckedPython -m face_matching.model_manager

if (-not $SkipGpuCheck) {
    Write-Host "Checking installation. A GPU error here means the selected driver/runtime is unavailable."
    Invoke-CheckedPython -m face_matching.diagnostics
} else {
    Write-Warning "GPU checks were skipped for cross-machine packaging. Run GPU诊断.bat on the target before use."
}
"Face Matching $Backend environment is ready." | Set-Content -LiteralPath $InstallMarker -Encoding ascii
