$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

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
                & py "-$Version" -m venv ".venv"
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
            throw "Python 3.11-3.13 was not found. Install 64-bit Python, then run install.bat again."
        }
        & python -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,14)"
        if ($LASTEXITCODE -ne 0) {
            throw "Python must be a 64-bit Python 3.11-3.13 installation."
        }
        & python -m venv ".venv"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
            throw "Failed to create the Python virtual environment."
        }
    }
}

& $VenvPython -c "import struct, sys; assert (3,11) <= sys.version_info[:2] < (3,14) and struct.calcsize('P') == 8"
if ($LASTEXITCODE -ne 0) {
    throw "The virtual environment must use 64-bit Python 3.11-3.13."
}

Write-Host "Installing GPU runtime and desktop dependencies..."
Invoke-CheckedPython -m pip install --upgrade pip setuptools wheel
Invoke-CheckedPython -m pip uninstall -y onnxruntime onnxruntime-directml
Invoke-CheckedPython -m pip install --upgrade --force-reinstall "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"
Invoke-CheckedPython -m pip install -e ".[dev]"
Invoke-CheckedPython -c "from face_matching.gpu import assert_gpu_available; print('GPU provider:', assert_gpu_available())"

Write-Host "Downloading research model weights (about 750 MB)..."
Write-Host "NOTICE: bundled download sources permit pretrained weights for non-commercial research only."
Invoke-CheckedPython -m face_matching.model_manager

Write-Host "Checking installation. A CUDA error here means the NVIDIA driver/runtime is unavailable."
Invoke-CheckedPython -m face_matching.diagnostics
