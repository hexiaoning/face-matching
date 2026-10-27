[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$ModelDirectory = "",
    [switch]$RunGpuSmoke
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Python) { $Python = Join-Path $ProjectRoot ".build-venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Create a Python 3.12 build environment with .[build], or pass -Python."
}
if (-not $ModelDirectory) { $ModelDirectory = Join-Path $ProjectRoot ".build-assets\video-analysis-models" }
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit $LASTEXITCODE" }
}
Set-Location -LiteralPath $ProjectRoot
$Version = (& $Python -c "from face_matching import __version__; print(__version__)").Trim()
$ReleaseName = "VideoAnalysis-v$Version-windows-x64"
$ReleaseRoot = Join-Path $ProjectRoot "dist"
$ReleaseDirectory = [IO.Path]::GetFullPath((Join-Path $ReleaseRoot $ReleaseName))
$ExpectedPrefix = $ProjectRoot.TrimEnd('\') + '\'
if (-not $ReleaseDirectory.StartsWith($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Release directory escapes the project workspace: $ReleaseDirectory"
}
$Archive = Join-Path $ReleaseRoot ($ReleaseName + ".zip")
$WorkPath = Join-Path $ProjectRoot ".build-assets\video-analysis-build"

Write-Host "[1/5] Running checks for v$Version..."
Invoke-CheckedPython -Arguments @("-m", "pytest")
Invoke-CheckedPython -Arguments @("-m", "face_matching.models", "download", "--person-vehicle", "--directory", $ModelDirectory)
Invoke-CheckedPython -Arguments @("-m", "face_matching.models", "verify", "--person-vehicle", "--directory", $ModelDirectory)

Write-Host "[2/5] Freezing standalone application, runtimes, and RT-DETRv2..."
$env:FACE_MATCHING_PROJECT_ROOT = $ProjectRoot
$env:FACE_MATCHING_BUNDLE_MODEL_DIR = [IO.Path]::GetFullPath($ModelDirectory)
try {
    Invoke-CheckedPython -Arguments @("-m", "PyInstaller", "--noconfirm", "--distpath", $ReleaseRoot,
        "--workpath", $WorkPath, (Join-Path $ProjectRoot "packaging\VideoAnalysis.spec"))
} finally {
    Remove-Item Env:FACE_MATCHING_PROJECT_ROOT -ErrorAction SilentlyContinue
    Remove-Item Env:FACE_MATCHING_BUNDLE_MODEL_DIR -ErrorAction SilentlyContinue
}
$Executable = Join-Path $ReleaseDirectory "VideoAnalysis.exe"
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Standalone EXE is missing" }
if ((Get-Item -LiteralPath $Executable).VersionInfo.ProductVersion -ne $Version) { throw "EXE version mismatch" }
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\start_video_analysis.cmd") -Destination (Join-Path $ReleaseDirectory "启动离线分析.cmd")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\VideoAnalysis-README.txt") -Destination (Join-Path $ReleaseDirectory "使用说明.txt")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "packaging\VideoAnalysis-NOTICES.txt") -Destination (Join-Path $ReleaseDirectory "THIRD_PARTY_NOTICES.txt")

Write-Host "[3/5] Auditing model payload and offline runtime libraries..."
Invoke-CheckedPython -Arguments @((Join-Path $ProjectRoot "scripts\audit_video_analysis_bundle.py"), $ReleaseDirectory)
$env:FACE_MATCHING_HOME = Join-Path $ProjectRoot ".build-assets\video-analysis-clean-home"
$env:QT_QPA_PLATFORM = "offscreen"
try {
    foreach ($Mode in @("check-runtime", "diagnose")) {
        if ($Mode -eq "diagnose" -and -not $RunGpuSmoke) { continue }
        $Report = Join-Path $ReleaseDirectory ($Mode + ".json")
        $Arguments = @("--$Mode", "--report", ('"{0}"' -f $Report))
        $Process = Start-Process -FilePath $Executable -ArgumentList $Arguments -Wait -PassThru -WindowStyle Hidden
        if ($Process.ExitCode -ne 0) {
            if (Test-Path -LiteralPath $Report) { Get-Content -LiteralPath $Report }
            throw "Packaged $Mode failed (exit $($Process.ExitCode))"
        }
        $Payload = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
        if (-not $Payload.ok -or $Payload.version -ne $Version -or $Payload.loaded_identity_modules.Count -ne 0) {
            throw "Invalid packaged runtime report"
        }
    }
} finally {
    Remove-Item Env:FACE_MATCHING_HOME -ErrorAction SilentlyContinue
    Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
}

Write-Host "[4/5] Recording build identity and creating complete ZIP..."
$BuildCommit = (git -C $ProjectRoot rev-parse HEAD).Trim()
@{ application="VideoAnalysis"; version=$Version; source_commit=$BuildCommit;
    created_at=[DateTime]::UtcNow.ToString("o"); artifact="complete-offline-zip" } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ReleaseDirectory "release-info.json") -Encoding UTF8
Invoke-CheckedPython -Arguments @((Join-Path $ProjectRoot "scripts\create_release_archive.py"), $ReleaseDirectory, $Archive)
Write-Host "[5/5] Complete offline ZIP ready: $Archive"
