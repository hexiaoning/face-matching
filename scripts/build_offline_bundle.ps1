[CmdletBinding()]
param(
    [ValidateSet("lvface-b", "auraface")]
    [string[]]$Profiles = @("lvface-b", "auraface"),
    [switch]$AcceptResearchWeights,
    [switch]$Clean,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildEnvironment = Join-Path $ProjectRoot ".build-venv"
$BuildPython = Join-Path $BuildEnvironment "Scripts\python.exe"
$ModelRoot = Join-Path $ProjectRoot ".build-assets\models"
$ReleaseRoot = Join-Path $ProjectRoot "dist"
$ReleaseName = "FaceMatching-v3.1.0-windows-x64"
$ReleaseDirectory = Join-Path $ReleaseRoot $ReleaseName
$Archive = Join-Path $ReleaseRoot ($ReleaseName + ".zip")

function Assert-SafeBuildTarget {
    param([Parameter(Mandatory = $true)][string]$Path)
    $FullPath = [IO.Path]::GetFullPath($Path)
    $Prefix = $ProjectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $FullPath.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a path outside the project: $FullPath"
    }
}

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $BuildPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

if ($Profiles.Count -eq 0) {
    throw "Select at least one model profile for the offline package."
}
if ($Profiles -contains "lvface-b" -and -not $AcceptResearchWeights) {
    throw "LVFace-B weights are restricted to non-commercial research. Rerun with -AcceptResearchWeights only after accepting that license, or build -Profiles auraface."
}

if ($Clean) {
    foreach ($Target in @($BuildEnvironment, (Join-Path $ProjectRoot ".build-assets"), (Join-Path $ProjectRoot "build"), $ReleaseRoot)) {
        Assert-SafeBuildTarget -Path $Target
        if (Test-Path -LiteralPath $Target) {
            Remove-Item -LiteralPath $Target -Recurse -Force
        }
    }
}

Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath $BuildPython -PathType Leaf)) {
    $Launcher = Get-Command "py" -ErrorAction SilentlyContinue
    $LauncherArguments = @("-3.12")
    if (-not $Launcher) {
        $Launcher = Get-Command "python" -ErrorAction SilentlyContinue
        $LauncherArguments = @()
    }
    if (-not $Launcher) {
        throw "64-bit Python 3.12 is required on the connected build machine."
    }
    & $Launcher.Source @LauncherArguments -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "The selected build interpreter must be 64-bit Python 3.12."
    }
    & $Launcher.Source @LauncherArguments -m venv $BuildEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated build environment."
    }
}

& $BuildPython -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "The existing build environment is not 64-bit Python 3.12. Run again with -Clean."
}

Write-Host "[1/5] Installing bounded CUDA/OpenVINO GPU runtime dependencies..." -ForegroundColor Cyan
Invoke-CheckedPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-CheckedPython -Arguments @("-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-directml")
Invoke-CheckedPython -Arguments @("-m", "pip", "install", "--upgrade", ($ProjectRoot + "[build]"))

Write-Host "[2/5] Downloading and SHA-256 checking models for the offline package..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $ModelRoot -Force | Out-Null
foreach ($Profile in $Profiles) {
    Invoke-CheckedPython -Arguments @("-m", "face_matching.models", "download", "--profile", $Profile, "--directory", $ModelRoot)
    Invoke-CheckedPython -Arguments @("-m", "face_matching.models", "verify", "--profile", $Profile, "--directory", $ModelRoot)
}

if (-not $SkipTests) {
    Write-Host "[3/5] Running unit tests before packaging..." -ForegroundColor Cyan
    Invoke-CheckedPython -Arguments @("-m", "pytest")
} else {
    Write-Warning "Unit tests were skipped by request."
}

Write-Host "[4/5] Building a self-contained Windows x64 portable directory..." -ForegroundColor Cyan
$env:FACE_MATCHING_PROJECT_ROOT = $ProjectRoot
$env:FACE_MATCHING_BUNDLE_MODEL_DIR = $ModelRoot
try {
    Invoke-CheckedPython -Arguments @("-m", "PyInstaller", "--noconfirm", "--clean", (Join-Path $ProjectRoot "packaging\FaceMatching.spec"))
}
finally {
    Remove-Item Env:FACE_MATCHING_PROJECT_ROOT -ErrorAction SilentlyContinue
    Remove-Item Env:FACE_MATCHING_BUNDLE_MODEL_DIR -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath (Join-Path $ReleaseDirectory "FaceMatching.exe") -PathType Leaf)) {
    throw "PyInstaller did not create the expected offline release directory."
}
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\安装并启动 Face Matching.cmd") `
    -Destination (Join-Path $ReleaseDirectory "安装并启动 Face Matching.cmd") -Force

Write-Host "[5/5] Creating a Zip64 archive and SHA-256 manifests..." -ForegroundColor Cyan
Invoke-CheckedPython -Arguments @((Join-Path $ProjectRoot "scripts\create_release_archive.py"), $ReleaseDirectory, $Archive)
Write-Host "Offline package ready: $Archive" -ForegroundColor Green
Write-Host "The target machine needs Windows 11 x64 and either a compatible NVIDIA or Intel GPU display driver." -ForegroundColor Green
