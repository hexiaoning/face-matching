[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) "release"),
    [switch]$KeepBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
$BuildRoot = Join-Path $ProjectRoot ".offline-build"
$BuildRootResolved = [System.IO.Path]::GetFullPath($BuildRoot)
$ExpectedBuildRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot ".offline-build"))
if ($BuildRootResolved -ne $ExpectedBuildRoot -or -not $BuildRootResolved.StartsWith([System.IO.Path]::GetFullPath($ProjectRoot))) {
    throw "Refusing to use an unexpected build directory: $BuildRootResolved"
}

$VenvPython = Join-Path $BuildRoot "venv\Scripts\python.exe"
$ModelDirectory = Join-Path $BuildRoot "models"
$DistDirectory = Join-Path $BuildRoot "dist"
$WorkDirectory = Join-Path $BuildRoot "pyinstaller"

function Invoke-BuildPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): python $($Arguments -join ' ')"
    }
}

Set-Location $ProjectRoot
if (Test-Path -LiteralPath $BuildRootResolved) {
    Remove-Item -LiteralPath $BuildRootResolved -Recurse -Force
}
New-Item -ItemType Directory -Path $BuildRootResolved | Out-Null

$Launcher = Get-Command py -ErrorAction SilentlyContinue
if (-not $Launcher) {
    throw "The online build computer needs the Windows py launcher and 64-bit Python 3.12."
}
& py -3.12 -c "import struct,sys; assert struct.calcsize('P') == 8 and sys.version_info[:2] == (3,12)"
if ($LASTEXITCODE -ne 0) {
    throw "The online build computer needs 64-bit Python 3.12."
}
& py -3.12 -m venv (Join-Path $BuildRoot "venv")
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
    throw "Failed to create the isolated build environment."
}

Write-Host "[1/5] Installing pinned-range release dependencies..." -ForegroundColor Cyan
Invoke-BuildPython -m pip install --upgrade pip setuptools wheel
Invoke-BuildPython -m pip uninstall -y onnxruntime onnxruntime-directml
Invoke-BuildPython -m pip install --upgrade --force-reinstall "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"
Invoke-BuildPython -m pip install ".[release]"

Write-Host "[2/5] Downloading and verifying both ONNX models..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $ModelDirectory | Out-Null
$PreviousModelDirectory = $env:FACE_MATCHING_MODEL_DIR
try {
    $env:FACE_MATCHING_MODEL_DIR = $ModelDirectory
    Invoke-BuildPython -m face_matching.model_manager
} finally {
    $env:FACE_MATCHING_MODEL_DIR = $PreviousModelDirectory
}

Write-Host "[3/5] Building the self-contained GPU application..." -ForegroundColor Cyan
$env:FACE_MATCHING_BUNDLE_MODELS = $ModelDirectory
try {
    Invoke-BuildPython -m PyInstaller --noconfirm --clean `
        --distpath $DistDirectory --workpath $WorkDirectory `
        (Join-Path $ProjectRoot "packaging\face_matching.spec")
} finally {
    Remove-Item Env:FACE_MATCHING_BUNDLE_MODELS -ErrorAction SilentlyContinue
}

$ApplicationDirectory = Join-Path $DistDirectory "FaceMatching"
if (-not (Test-Path -LiteralPath (Join-Path $ApplicationDirectory "FaceMatching.exe"))) {
    throw "PyInstaller did not produce FaceMatching.exe."
}
Copy-Item -LiteralPath (Join-Path $ProjectRoot "offline\Start Face Matching.bat") -Destination $ApplicationDirectory
Copy-Item -LiteralPath (Join-Path $ProjectRoot "offline\GPU Diagnostics.bat") -Destination $ApplicationDirectory
Copy-Item -LiteralPath (Join-Path $ProjectRoot "offline\OFFLINE_DEPLOYMENT.txt") -Destination $ApplicationDirectory
Invoke-BuildPython -m pip freeze | Set-Content -LiteralPath (Join-Path $ApplicationDirectory "PYTHON_PACKAGES.txt") -Encoding UTF8

Write-Host "[4/5] Writing the per-file integrity manifest..." -ForegroundColor Cyan
$ManifestPath = Join-Path $ApplicationDirectory "SHA256SUMS.txt"
Get-ChildItem -LiteralPath $ApplicationDirectory -Recurse -File |
    Where-Object { $_.FullName -ne $ManifestPath } |
    Sort-Object FullName |
    ForEach-Object {
        $Relative = [System.IO.Path]::GetRelativePath($ApplicationDirectory, $_.FullName).Replace('\', '/')
        $Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$Hash  $Relative"
    } | Set-Content -LiteralPath $ManifestPath -Encoding ASCII

Write-Host "[5/5] Creating the offline ZIP..." -ForegroundColor Cyan
$OutputResolved = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputResolved -Force | Out-Null
$ArchivePath = Join-Path $OutputResolved "FaceMatching-0.2.2-win64-cuda-offline.zip"
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}
Compress-Archive -LiteralPath $ApplicationDirectory -DestinationPath $ArchivePath -CompressionLevel Optimal
$ArchiveHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath ($ArchivePath + ".sha256") -Encoding ASCII -Value "$ArchiveHash  $([System.IO.Path]::GetFileName($ArchivePath))"

if (-not $KeepBuild) {
    Remove-Item -LiteralPath $BuildRootResolved -Recurse -Force
}
Write-Host "Offline package: $ArchivePath" -ForegroundColor Green
Write-Host "Target requires only Windows, a supported NVIDIA GPU, and its display driver." -ForegroundColor Green
