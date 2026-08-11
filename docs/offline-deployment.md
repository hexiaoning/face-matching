# 完全离线部署

`v2.4` 的正式交付物是 PyInstaller `onedir` 便携包。联网构建机负责下载和固定哈希校验；目标机不运行 `pip`、不下载模型，也不要求预装 Python、CUDA Toolkit、cuDNN 或 OpenVINO Toolkit。

## 构建机

要求 Windows 11 x64、64 位 Python 3.12 和互联网连接。在 PowerShell 中运行：

```powershell
.\scripts\build_offline_bundle.ps1 -Clean -AcceptResearchWeights
```

默认同时包含 `lvface-b` 和 `auraface`，因此 GUI 中可离线切换。若没有接受 LVFace 非商业研究权重许可，只构建可商用模型档：

```powershell
.\scripts\build_offline_bundle.ps1 -Clean -Profiles auraface
```

输出位于 `dist\FaceMatching-v2.4.0-windows-x64.zip`，旁边的 `.sha256` 文件校验整个压缩包；包内 `SHA256SUMS.txt` 校验每个文件。包中包含：

- Python 3.12 运行时和应用代码；
- PySide6、OpenCV、NumPy；
- `onnxruntime-gpu` 及其 CUDA 12/cuDNN/cuBLAS 运行 DLL；
- OpenVINO Intel GPU 运行库；
- 构建时选择且经过 SHA-256 校验的 SCRFD/LVFace/AuraFace ONNX 模型；
- 离线自检脚本和第三方许可说明。

## 目标机

目标机要求 Windows 11 x64，以及 NVIDIA RTX 或 Intel UHD/Iris/Arc GPU 和对应显卡驱动。

1. 用 `.sha256` 核对 ZIP，解压到本机目录。
2. 在 PowerShell 运行 `verify_offline.ps1`。它会自动发现包内 profile，逐模型校验哈希，并在自动选中的 CUDA/OpenVINO GPU 上执行真实推理；任何 CPU fallback 都会直接失败。
3. 双击 `FaceMatching.exe`。

人员数据库和录入照片默认写入 `%LOCALAPPDATA%\FaceMatching`，应用升级不会覆盖。需要把数据放到加密盘时，在启动前设置 `FACE_MATCHING_HOME`。

## 交付检查

- 在一台禁网且从未安装 Python/CUDA Toolkit 的目标机上解压验证。
- `verify_offline.ps1` 发现的所有 profile 均返回 `"ok": true`。
- 使用目标摄像头的正负样本校准阈值，记录 FAR/TAR，而不是把余弦分数当概率。
- 对数据目录启用 BitLocker/EFS、最小权限、审计和保留期策略。
