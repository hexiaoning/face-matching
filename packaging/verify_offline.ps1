[CmdletBinding()]
param(
    [ValidateSet("lvface-b", "auraface")]
    [string[]]$Profiles = @()
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Executable = Join-Path $AppRoot "FaceMatching.exe"
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "FaceMatching.exe is missing from the offline bundle."
}

if ($Profiles.Count -eq 0) {
    $ModelRoots = @((Join-Path $AppRoot "_internal\models"), (Join-Path $AppRoot "models"))
    foreach ($ModelRoot in $ModelRoots) {
        if (Test-Path -LiteralPath (Join-Path $ModelRoot "LVFace-B_Glint360K.onnx") -PathType Leaf) {
            $Profiles += "lvface-b"
        }
        if (Test-Path -LiteralPath (Join-Path $ModelRoot "glintr100.onnx") -PathType Leaf) {
            $Profiles += "auraface"
        }
    }
    $Profiles = @($Profiles | Select-Object -Unique)
}
if ($Profiles.Count -eq 0) {
    throw "No supported offline recognizer model was found in the bundle."
}

foreach ($Profile in $Profiles) {
    $Report = Join-Path ([IO.Path]::GetTempPath()) ("face-matching-{0}-{1}.json" -f $Profile, [guid]::NewGuid().ToString("N"))
    try {
        Write-Host "Verifying model hashes and real CUDA/OpenVINO GPU inference: $Profile" -ForegroundColor Cyan
        $Arguments = @("--diagnose", "--profile", $Profile, "--report", ('"{0}"' -f $Report))
        $Process = Start-Process -FilePath $Executable -ArgumentList $Arguments -Wait -PassThru -WindowStyle Hidden
        if ($Process.ExitCode -ne 0) {
            if (Test-Path -LiteralPath $Report) {
                Get-Content -LiteralPath $Report
            }
            throw "Offline GPU verification failed for $Profile (exit $($Process.ExitCode))."
        }
        Get-Content -LiteralPath $Report
    }
    finally {
        Remove-Item -LiteralPath $Report -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Offline bundle is complete and GPU inference is ready." -ForegroundColor Green
