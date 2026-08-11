# Face Matching

Windows 11 桌面端监控视频人脸识别。人员库保存“姓名 + 身份证号 + 1～多张照片”，界面支持本地视频、摄像头、RTSP/RTMP 视频流，识别结果会在视频上标注并记录。`v2.3` 提供包含运行库和模型的完全离线便携包构建流程。

> 这是 **GPU-only** 应用。检测和人脸特征网络只创建 `CUDAExecutionProvider`，并禁用 ONNX Runtime CPU fallback。CUDA/NVIDIA 驱动或 GPU 模型加载不可用时，程序会明确报错并退出。视频解码、跟踪管理、SQLite 和 GUI 等非神经网络工作仍由 CPU 执行。

> 这是部署在受控环境中的**内部系统，不需要脱敏功能**。程序不遮罩身份证号，也不清理视频源地址中的用户名或密码；人员表、识别结果和事件日志均保留并显示完整原始值。部署方应通过内部账号权限、主机访问控制和磁盘加密保护这些数据。

## 方案

```text
视频/摄像头
   → SCRFD-10G 人脸检测 + 5 点对齐 (CUDA)
   → 模糊/姿态/尺寸/照明质量门控
   → LVFace-B 特征 + 水平镜像测试增强 (CUDA)
   → 位置 + 运动预测 + 外观特征的人脸轨迹关联
   → 同一轨迹 Top-K 高质量帧加权聚合 + 离群身份帧剔除
   → 每人多照片模板（最佳样本 + Top-3 质量加权 + 人员中心）
   → 开集判定（相似度阈值 + 第一/第二候选差值 + 多帧共识）
```

选型依据和限制见 [docs/architecture.md](docs/architecture.md)。默认的 LVFace-B 是 ICCV 2025 Highlight 的开源实现，官方公布的 Glint360K 模型在 IJB-C TAR@FAR=1e-4 为 97.70%。这不等于未经校准的实际监控准确率，部署前必须用现场数据调阈值。

## 支持环境

- Windows 11 64 位
- NVIDIA RTX GPU（目标机型：RTX 4070 SUPER 12 GB）
- 较新的 NVIDIA 显卡驱动
- 不需要预先安装 CUDA Toolkit/cuDNN；安装脚本通过官方 `onnxruntime-gpu[cuda,cudnn]` wheel 安装 CUDA 12/cuDNN 运行 DLL

## 正式交付：目标机完全离线

在联网的 Windows x64 构建机上运行：

```powershell
.\scripts\build_offline_bundle.ps1 -Clean -AcceptResearchWeights
```

构建脚本会先跑测试，下载并校验固定 SHA-256 的模型，然后把 Python、PySide6、OpenCV、ONNX Runtime GPU、CUDA/cuDNN/cuBLAS DLL 和两套模型一并打入 `dist\FaceMatching-v2.3.0-windows-x64.zip`。目标机不需要 Python、CUDA Toolkit 或互联网；只需预装兼容的 NVIDIA 显卡驱动，解压后先运行 `verify_offline.ps1`，再双击 `FaceMatching.exe`。详细步骤见 [离线部署文档](docs/offline-deployment.md)。

> NVIDIA 显卡驱动是与机器/系统绑定的设备组件，不随应用包分发；其余应用运行依赖和模型均包含在离线包中。包含 LVFace-B 权重前必须接受其非商业研究限制；商业部署可只构建 `auraface`。

## 开发机联网安装与运行

1. 安装/更新 [NVIDIA 显卡驱动](https://www.nvidia.com/Download/index.aspx)。
2. 双击 `install.cmd`。脚本会创建隔离的 `.venv`、安装 GUI 与 CUDA/cuDNN 运行库，然后执行 GPU 硬校验。如果没有合适的 Python，Windows 11 上会先通过 `winget` 安装用户级 Python 3.12。该路径用于开发/评估，需要联网，不是目标机部署方式。
3. 双击 `run.cmd`。
4. 开发模式首次运行会询问是否下载并校验模型；离线便携版绝不会联网补模型。

也可以在 PowerShell 中运行：

```powershell
.\install.ps1
.\run.ps1
```

GPU 自检（不会回退 CPU）：

```powershell
.\.venv\Scripts\python.exe -m face_matching --check-gpu
```

模型哈希 + 两个模型真实 CUDA 推理诊断：

```powershell
.\.venv\Scripts\python.exe -m face_matching --diagnose --profile lvface-b --report diagnostics.json
```

## 鼠标操作

1. 打开“人员库”，点“录入人员”，填姓名和身份证号，选择 1～多张照片。建议同时提供清晰正面、左右轻度侧脸照片。
2. 在“视频识别”中选视频，或输入 `0` 打开本机摄像头，或输入 RTSP/RTMP 地址。
3. 点“开始识别”。绿色为已确认人员，黄色为采集中/陌生人。内部系统会在表格中显示完整身份证号。
4. 在“设置”中调相似度、质量、检测尺寸等参数。离线 GUI 只显示包内已有模型；改识别模型后需重启，并在人员库中点“重建当前模型特征”。

## 模型档位与许可证

| 档位 | 用途 | 权重许可 | 大小 |
| --- | --- | --- | --- |
| `lvface-b`（默认） | 优先准确率，研究/评估 | **仅非商业研究**；代码 MIT | 约 472 MB（含检测器） |
| `auraface` | 需要宽松权重许可的应用 | Apache-2.0 | 约 278 MB（含检测器） |

程序下载时校验固定文件大小和 SHA-256。使用 LVFace 做商业、安防或生产部署前，必须先获得相应模型权重授权，或在设置中改用 AuraFace 并重建特征。

## 数据位置

默认保存在 `%LOCALAPPDATA%\FaceMatching`：

- `models/`：下载的 ONNX 模型
- `data/faces.db`：SQLite 人员库、特征和识别事件
- `data/face_images/`：录入照片副本
- `config.json`：界面设置

可用 `FACE_MATCHING_HOME` 环境变量改变数据目录。身份证号和人脸特征属于敏感个人信息；生产部署应使用 BitLocker/EFS 或组织的密钥系统加密数据目录，并设置访问权限、保留期和审计。

## 开发验证

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall -q src
```

纯逻辑测试可在无 GPU 的开发机上执行；真实模型启动必须通过 CUDA 硬校验。
