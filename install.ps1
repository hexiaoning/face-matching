[CmdletBinding()]
param(
    [switch]$SkipGpuCheck
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"

function Find-Python {
    $Candidates = @("python", "python3")
    if ($env:LOCALAPPDATA) {
        $Candidates += @(
            (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"),
            (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
            (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
            (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe")
        )
    }
    foreach ($Candidate in $Candidates) {
        $Command = Get-Command $Candidate -ErrorAction SilentlyContinue
        if (-not $Command) { continue }
        & $Command.Source -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) and sys.maxsize > 2**32 else 1)"
        if ($LASTEXITCODE -eq 0) { return $Command.Source }
    }
    return $null
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

Set-Location $ProjectRoot
$Python = Find-Python
if (-not $Python) {
    $Winget = Get-Command "winget" -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "Python 3.10-3.13 64-bit was not found, and winget is unavailable for automatic installation."
    }
    Write-Host "Python was not found. Installing 64-bit Python 3.12 for the current user..." -ForegroundColor Cyan
    & $Winget.Source install --exact --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Automatic Python installation failed (winget exit $LASTEXITCODE)."
    }
    $Python = Find-Python
    if (-not $Python) {
        throw "Python was installed but could not be located. Sign out and back in, then rerun install.cmd."
    }
}
Write-Host "[1/4] Creating isolated Python environment..." -ForegroundColor Cyan
if (-not (Test-Path $VenvPython)) {
    Invoke-CheckedPython -Executable $Python -Arguments @("-m", "venv", $VenvRoot)
}

Write-Host "[2/4] Updating installer..." -ForegroundColor Cyan
Invoke-CheckedPython -Executable $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

Write-Host "[3/4] Installing GUI, ONNX Runtime GPU, and bundled CUDA/cuDNN runtime DLLs..." -ForegroundColor Cyan
# Avoid two distributions racing to provide the same onnxruntime Python package.
Invoke-CheckedPython -Executable $VenvPython -Arguments @("-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-directml")
Invoke-CheckedPython -Executable $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "-e", $ProjectRoot)

if (-not $SkipGpuCheck) {
    Write-Host "[4/4] Verifying CUDA-only inference..." -ForegroundColor Cyan
    & $VenvPython -m face_matching --check-gpu
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA GPU verification failed. CPU fallback is intentionally disabled. Update the NVIDIA driver and rerun install.cmd."
    }
} else {
    Write-Warning "GPU verification was skipped. The application will still refuse to start without CUDA."
}

Write-Host "Ready. Double-click run.cmd to start." -ForegroundColor Green
