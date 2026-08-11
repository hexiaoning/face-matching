[CmdletBinding()]
param(
    [string]$ReleaseDirectory = "",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $ReleaseDirectory) {
    $ReleaseDirectory = Join-Path $ProjectRoot "dist\FaceMatching-v3.4.0-windows-x64"
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $ProjectRoot "dist"
}
$ReleaseDirectory = [IO.Path]::GetFullPath($ReleaseDirectory)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (-not (Test-Path -LiteralPath (Join-Path $ReleaseDirectory "FaceMatching.exe") -PathType Leaf)) {
    throw "Portable release is missing: $ReleaseDirectory"
}

$Candidates = @(
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
)
$Compiler = $Candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
if (-not $Compiler) {
    throw "Inno Setup 6 was not found. Install it on the connected build machine with: winget install --id JRSoftware.InnoSetup -e"
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$env:FACE_MATCHING_INSTALLER_SOURCE = $ReleaseDirectory
$env:FACE_MATCHING_INSTALLER_OUTPUT = $OutputDirectory
try {
    & $Compiler (Join-Path $ProjectRoot "packaging\FaceMatching.iss")
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:FACE_MATCHING_INSTALLER_SOURCE -ErrorAction SilentlyContinue
    Remove-Item Env:FACE_MATCHING_INSTALLER_OUTPUT -ErrorAction SilentlyContinue
}
$Installer = Join-Path $OutputDirectory "FaceMatching-v3.4.0-Setup.exe"
if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
    throw "Installer was not created: $Installer"
}
$InstallerHash = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
$ChecksumFile = "$Installer.sha256"
$ChecksumLine = "$InstallerHash  $(Split-Path -Leaf $Installer)`n"
[IO.File]::WriteAllText($ChecksumFile, $ChecksumLine, [Text.UTF8Encoding]::new($false))
Write-Output $Installer
Write-Output $ChecksumFile
