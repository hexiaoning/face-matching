[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [switch]$SkipModelDownload,
    [switch]$SkipGpuSelfTest
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SetupScript = Join-Path $PSScriptRoot "setup.ps1"

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $ProjectRoot "offline_dist"
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$BundleDirectory = Join-Path $OutputDirectory "FaceMatching"
$ArchivePath = Join-Path $OutputDirectory "FaceMatching-offline-win64.zip"
$BuildRoot = Join-Path $ProjectRoot "build\offline-pyinstaller"
$SpecRoot = Join-Path $ProjectRoot "build\offline-spec"
$ModelStage = Join-Path $ProjectRoot "build\offline-models"

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): python $($Arguments -join ' ')"
    }
}

Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Preparing the GPU build environment..." -ForegroundColor Cyan
    & $SetupScript
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
        throw "The build environment could not be created."
    }
}

Write-Host "Installing the reproducible bundle builder..." -ForegroundColor Cyan
Invoke-CheckedPython -m pip install -e ".[build]"
if (-not $SkipModelDownload) {
    Write-Host "Downloading/verifying both model files on the BUILD machine..." -ForegroundColor Cyan
    Invoke-CheckedPython -m face_matching.model_manager
}

$ModelJson = & $VenvPython -c "import json; from face_matching.model_manager import required_models; print(json.dumps([str(p) for p in required_models()]))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not resolve model paths."
}
$ModelPaths = $ModelJson | ConvertFrom-Json
foreach ($ModelPath in $ModelPaths) {
    if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
        throw "Required model is missing: $ModelPath"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory, $BuildRoot, $SpecRoot | Out-Null
if (Test-Path -LiteralPath $ModelStage) {
    Remove-Item -LiteralPath $ModelStage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $ModelStage | Out-Null
$StagedDetector = Join-Path $ModelStage "scrfd_10g_bnkps.onnx"
$StagedRecognizer = Join-Path $ModelStage "LVFace-B_Glint360K.onnx"
Copy-Item -LiteralPath $ModelPaths[0] -Destination $StagedDetector
Copy-Item -LiteralPath $ModelPaths[1] -Destination $StagedRecognizer
if (Test-Path -LiteralPath $BundleDirectory) {
    Remove-Item -LiteralPath $BundleDirectory -Recurse -Force
}

$Launcher = Join-Path $ProjectRoot "packaging\launcher.py"
$PyInstallerArguments = @(
    "-m", "PyInstaller",
    "--noconfirm", "--clean", "--windowed", "--onedir", "--noupx",
    "--name", "FaceMatching",
    "--contents-directory", "_internal",
    "--paths", (Join-Path $ProjectRoot "src"),
    "--collect-all", "onnxruntime",
    "--distpath", $OutputDirectory,
    "--workpath", $BuildRoot,
    "--specpath", $SpecRoot,
    "--add-data", "$($StagedDetector):models",
    "--add-data", "$($StagedRecognizer):models",
    $Launcher
)
Write-Host "Freezing Python, GUI, OpenCV and ONNX Runtime..." -ForegroundColor Cyan
Invoke-CheckedPython @PyInstallerArguments

$InternalDirectory = Join-Path $BundleDirectory "_internal"
$CudaDirectory = Join-Path $InternalDirectory "cuda_dlls"
New-Item -ItemType Directory -Force -Path $CudaDirectory | Out-Null
$SitePackages = (& $VenvPython -c "import site; print(site.getsitepackages()[0])").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not resolve site-packages."
}
$NvidiaRoot = Join-Path $SitePackages "nvidia"
if (-not (Test-Path -LiteralPath $NvidiaRoot -PathType Container)) {
    throw "Bundled CUDA libraries are missing. Reinstall onnxruntime-gpu[cuda,cudnn]."
}
$CudaDlls = Get-ChildItem -LiteralPath $NvidiaRoot -Recurse -File -Filter "*.dll"
if (-not $CudaDlls) {
    throw "No CUDA/cuDNN DLLs were found below $NvidiaRoot"
}
foreach ($Dll in $CudaDlls) {
    Copy-Item -LiteralPath $Dll.FullName -Destination (Join-Path $CudaDirectory $Dll.Name) -Force
}
foreach ($Pattern in @("cudart64*.dll", "cublas64*.dll", "cudnn64*.dll")) {
    if (-not (Get-ChildItem -LiteralPath $CudaDirectory -File -Filter $Pattern)) {
        throw "Offline bundle is incomplete; missing $Pattern"
    }
}

Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\start_offline.bat") -Destination (Join-Path $BundleDirectory "启动.bat")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\diagnose_offline.bat") -Destination (Join-Path $BundleDirectory "GPU诊断.bat")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\OFFLINE_README.txt") -Destination (Join-Path $BundleDirectory "离线部署说明.txt")

if (-not $SkipGpuSelfTest) {
    Write-Host "Running real detector + recognizer inference from the frozen bundle..." -ForegroundColor Cyan
    $DiagnosticPath = Join-Path $BundleDirectory "diagnostics-build.json"
    $Process = Start-Process -FilePath (Join-Path $BundleDirectory "FaceMatching.exe") `
        -ArgumentList @("--diagnose-file", ('"{0}"' -f $DiagnosticPath)) `
        -Wait -PassThru -WindowStyle Hidden
    if ($Process.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $DiagnosticPath) {
            Get-Content -Raw -LiteralPath $DiagnosticPath | Write-Host
        }
        throw "Frozen-bundle GPU self-test failed with exit code $($Process.ExitCode)."
    }
    Remove-Item -LiteralPath $DiagnosticPath -Force
}

$Versions = & $VenvPython -m pip list --format=freeze --disable-pip-version-check
if ($LASTEXITCODE -ne 0) {
    throw "Could not create the dependency version inventory."
}
$Versions | Set-Content -LiteralPath (Join-Path $BundleDirectory "THIRD_PARTY_VERSIONS.txt") `
    -Encoding utf8
$ManifestPath = Join-Path $BundleDirectory "bundle-manifest.sha256"
$ManifestLines = foreach ($File in Get-ChildItem -LiteralPath $BundleDirectory -Recurse -File) {
    if ($File.FullName -eq $ManifestPath) {
        continue
    }
    $Relative = $File.FullName.Substring($BundleDirectory.TrimEnd("\").Length + 1).Replace("\", "/")
    $Hash = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash *$Relative"
}
$ManifestLines | Set-Content -LiteralPath $ManifestPath -Encoding ascii

if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory(
    $BundleDirectory,
    $ArchivePath,
    [IO.Compression.CompressionLevel]::Optimal,
    $false
)
Write-Host "Offline folder: $BundleDirectory" -ForegroundColor Green
Write-Host "Offline archive: $ArchivePath" -ForegroundColor Green
