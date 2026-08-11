$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" }
if (-not $gpu) {
    throw "未检测到 NVIDIA GPU。本项目禁止 CPU 推理，请先安装 NVIDIA 显卡和官方驱动。"
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    Write-Host "正在通过 winget 安装 uv..."
    winget install --id astral-sh.uv --exact --accept-package-agreements --accept-source-agreements
    $uvCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe"),
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe")
    )
    $uvPath = $uvCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $uvPath) {
        throw "uv 已安装但当前终端尚未刷新 PATH。请关闭窗口后再次双击 install.bat。"
    }
} else {
    $uvPath = $uvCommand.Source
}

Write-Host "正在准备 Python 3.12..."
& $uvPath python install 3.12

Write-Host "正在安装 GUI、ONNX Runtime GPU、CUDA 和 cuDNN 运行库..."
& $uvPath sync

Write-Host "正在验证 GPU 版 ONNX Runtime..."
& $uvPath run --no-sync python -c "from face_matching.gpu import load_onnxruntime_gpu; o=load_onnxruntime_gpu(); print('可用执行器:', o.get_available_providers())"

Write-Host "安装完成。双击 run.bat 启动。首次启动可用鼠标确认下载研究模型。" -ForegroundColor Green
Read-Host "按回车键关闭"
