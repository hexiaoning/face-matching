# Face Matching

Windows 11 桌面端目标人物视频检索。直接给定一张目标人物照片，即可检查本地监控视频、摄像头或 RTSP/RTMP 流中此人是否出现。`v3.3` 将一对一目标检索与人员库的一对多识别分开：目标检索保留每条轨迹的多个独立帧证据，并输出视频内时间、候选人脸缩略图以及“自动确认 / 待复核”判定。

> 这是 **GPU-only** 应用。NVIDIA 使用 ONNX Runtime CUDA，Intel 核显/Arc 使用 OpenVINO GPU FP32；两种后端都禁止神经网络回退 CPU。`auto` 优先 CUDA，再选 OpenVINO。视频解码、跟踪管理、SQLite 和 GUI 等非神经网络工作仍由 CPU 执行。

> 这是部署在受控环境中的**内部系统，不需要脱敏功能**。程序不遮罩身份证号，也不清理视频源地址中的用户名或密码；人员表、识别结果和事件日志均保留并显示完整原始值。部署方应通过内部账号权限、主机访问控制和磁盘加密保护这些数据。

## 方案

```text
目标照片
   → 紧裁照片上下文补全检测 + 多个轻微对齐模板

视频/摄像头
   → SCRFD-10G 人脸检测 + 5 点对齐 (CUDA / OpenVINO GPU)
   → 模糊/姿态/尺寸/照明质量门控
   → LVFace-B 特征 + 水平镜像测试增强 (CUDA / OpenVINO GPU)
   → 位置 + 运动预测 + 外观特征的人脸轨迹关联
   → 每帧与目标多个参考视角做一对一比对
   → 多个独立视频帧共同支持才自动确认；弱证据进入人工复核
   → 输出候选人脸、视频时间点、综合分、最佳帧分和支持帧数
```

选型依据和限制见 [docs/architecture.md](docs/architecture.md)。默认的 LVFace-B 是 ICCV 2025 Highlight 的开源实现，官方公布的 Glint360K 模型在 IJB-C TAR@FAR=1e-4 为 97.70%。这不等于未经校准的实际监控准确率，部署前必须用现场数据调阈值。

## 支持环境

- Windows 11 64 位
- NVIDIA RTX GPU（目标机型：RTX 4070 SUPER 12 GB）或 Intel UHD/Iris/Arc GPU
- 对应的较新 NVIDIA/Intel 显卡驱动
- 不需要预先安装 CUDA Toolkit、cuDNN 或 OpenVINO Toolkit

## 正式交付：目标机完全离线

在联网的 Windows x64 构建机上运行：

```powershell
.\scripts\build_offline_bundle.ps1 -Clean -AcceptResearchWeights
```

构建脚本会先跑测试，下载并校验固定 SHA-256 的模型，然后把 Python、PySide6、OpenCV、CUDA/OpenVINO GPU 运行库和两套模型一并打入 `dist\FaceMatching-v3.3.0-windows-x64.zip`。目标机不需要 Python、CUDA Toolkit、OpenVINO Toolkit 或互联网；只需预装对应显卡驱动。

目标机可直接双击 `FaceMatching-v3.3.0-Setup.exe` 安装；也可解压便携 ZIP 后双击 `安装并启动 Face Matching.cmd`，自动校验完整性、GPU 和模型后启动。两种方式都不会调用 `winget` 或 `pip`。

> NVIDIA 显卡驱动是与机器/系统绑定的设备组件，不随应用包分发；其余应用运行依赖和模型均包含在离线包中。包含 LVFace-B 权重前必须接受其非商业研究限制；商业部署可只构建 `auraface`。

## 开发机联网安装与运行

1. 安装/更新 NVIDIA 或 Intel 显卡驱动。
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

模型哈希 + 两个模型真实 GPU 推理诊断：

```powershell
.\.venv\Scripts\python.exe -m face_matching --diagnose --profile lvface-b --report diagnostics.json
```

## 鼠标操作

1. 在“目标检索”选择一张只包含目标人物的照片。普通照片和紧裁大头照均可；系统会建立多个轻微对齐的参考模板。
2. 选择本地视频，或输入 `0` 打开本机摄像头，或输入 RTSP/RTMP 地址。本地视频默认尽可能快地扫描，不按原时长等待。
3. 点“开始检索”。绿色表示至少多个独立帧达到确认阈值；橙色表示分数值得人工复核但证据不足。右侧保留候选人脸、视频时间点和支持帧数。
4. 在“设置”中可分别调整自动确认阈值、待复核阈值和最少支持帧。默认参数是可用起点，正式部署仍必须用现场正负样本校准。

原有“人员库”仍保留，供需要多人员资料管理或一对多识别的场景使用；它不再是单目标视频检索的前置步骤。

镜像 TTA 状态会写入特征 `model_id`（`-tta` / `-single`）。两种特征空间不会混用；从 v2.3 升级后必须重建一次人员库特征。

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

纯逻辑测试可在无 GPU 的开发机上执行；真实模型启动必须通过 CUDA 或 OpenVINO GPU 硬校验。
