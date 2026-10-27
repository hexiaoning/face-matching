# Compatibility entry point: every build now contains the complete application.
[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$ModelDirectory = "",
    [ValidateSet("lvface-b", "auraface")]
    [string[]]$Profiles = @("lvface-b", "auraface"),
    [switch]$AcceptResearchWeights,
    [switch]$RunGpuSmoke
)
& (Join-Path $PSScriptRoot "build_offline_bundle.ps1") @PSBoundParameters
