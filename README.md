# Face Matching

面向监控视频的本地人脸检索桌面程序。人员库保存“姓名 + 身份证号 + 1～多张照片”，支持鼠标选择本地视频、USB 摄像头或 RTSP/HTTP 视频流，并在画面中显示轨迹、识别结果、完整身份证号和相似度。

本项目**强制使用 NVIDIA GPU**。CUDA Execution Provider 不存在、模型无法在 CUDA 上加载或首次 GPU 推理失败时，程序会明确报错退出，绝不静默降级到 CPU。

## 技术方案

这里没有对所有摄像头和人群都通用的单一“SOTA”。本实现选择了截至 2026 年仍具备官方权重、ONNX 推理路径和良好困难场景指标的可落地组合：

- **SCRFD-10G**：检测人脸和五点关键点。SCRFD 在 WIDER FACE 困难集兼顾精度与速度，适合视频逐帧检测。
- **五点仿射对齐**：把偏转、尺度不同的人脸统一到 LVFace 使用的 `112×112` ArcFace 模板。
- **LVFace-B / Glint360K**：ICCV 2025 Highlight 的 Vision Transformer 人脸特征模型。官方报告在 IJB-C 上达到 TAR `90.06% @ FAR=1e-6`、`97.70% @ FAR=1e-4`，并提供 ONNX 权重。
- **质量门控**：综合检测置信度、人脸像素尺寸、拉普拉斯清晰度与五点姿态对称性。极差帧只显示、不进入身份判断。
- **轨迹级聚合**：同一人脸轨迹保留多帧特征，按质量和时间加权后再匹配。模糊或非正面帧不会单独决定身份。
- **镜像测试增强（TTA）**：录入照和视频探针同时融合原图/水平翻转特征，增强姿态变化下的稳定性；多张人脸合并成 GPU batch，减少调用开销。
- **鲁棒轨迹模板**：以轨迹内 medoid 为中心剔除身份不一致的离群帧，避免一次错轨或坏帧污染整个轨迹。
- **开放集判定**：不仅要求最高余弦相似度超过阈值，还要求第一名和第二名之间有足够 margin，并用两次真实的新特征结果做时序确认。
- **多参考照人员模板**：每个人的多个样本先在人员级别打分，清晰样本权重更高，减少单张照片的偶然性。

相关官方资料：

- [LVFace 官方代码、权重和指标](https://github.com/bytedance/LVFace)
- [LVFace ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/html/You_LVFace_Progressive_Cluster_Optimization_for_Large_Vision_Models_in_Face_ICCV_2025_paper.html)
- [SCRFD 官方实现和 WIDER FACE 指标](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [AdaFace 低质量人脸研究](https://github.com/mk-minchul/AdaFace)（方案比较参考）
- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

没有加入生成式“人脸修复/超分辨率”：这类模型可以让画面看起来更清晰，但也可能生成原图不存在的身份细节。检索系统更安全的做法是拒绝极差帧，并聚合同一轨迹中的较好帧。

## 支持环境

目标设备：

| 项目 | 配置 |
| --- | --- |
| 设备 | 华硕 ROG G22CH 台式机 |
| CPU | Intel i7-14700KF |
| 显卡 | RTX 4070 SUPER 12GB |
| 内存 | 64GB |
| 系统 | Windows 11 64 位 |
| Python | 源码安装需 64 位 Python 3.11～3.13；离线发布包无需 Python |

设备无需预装完整 CUDA Toolkit。源码安装通过 `onnxruntime-gpu[cuda,cudnn]` 安装 CUDA 12 / cuDNN 运行库；离线发布包已包含这些用户态 DLL。两种方式都仍需目标机安装支持该显卡的 NVIDIA 显示驱动，因为驱动提供的 `nvcuda.dll` 不能随应用分发。

## 离线部署（交付目标机时使用）

在一台**可联网的 Windows 构建机**上安装 64 位 Python 3.12，然后双击：

```text
build_offline_bundle.bat
```

脚本会在隔离环境中下载并校验依赖和约 750 MB 模型，使用 PyInstaller 生成 `release/FaceMatching-0.2.2-win64-cuda-offline.zip`，同时生成 ZIP 的 `.sha256`。发布包内包含：

- Python 运行时、PySide6/Qt、OpenCV、NumPy；
- ONNX Runtime GPU 与 pip 提供的 CUDA/cuDNN/CUBLAS 运行 DLL；
- SCRFD-10G 和 LVFace-B 两个经过 SHA-256 校验的 ONNX 模型；
- GPU 自检、启动脚本、依赖清单和包内逐文件哈希清单。

目标机不需要联网，不需要安装 Python、CUDA Toolkit 或 cuDNN。把整个 ZIP 解压到本地目录，先双击 `GPU Diagnostics.bat`，通过后再双击 `Start Face Matching.bat`。唯一的外部前提是 NVIDIA 显示驱动可用（`nvidia-smi` 能正常运行）。

## 源码安装与启动（开发机）

1. 安装最新 NVIDIA 显卡驱动和 64 位 Python 3.11～3.13。
2. 双击 `install.bat`。此流程**需要联网**，会创建 `.venv`、安装桌面与 CUDA 依赖、下载约 750 MB 模型，并执行两个模型的真实 GPU 推理自检。
3. 双击 `start.bat`。

安装失败时不会留下“看似成功”的状态。可在 PowerShell 中重新诊断：

```powershell
.\.venv\Scripts\python.exe -m face_matching.diagnostics
```

诊断成功时 `gpu_ready` 和 `inference_ready` 都应为 `true`，`active_provider` 应为 `CUDAExecutionProvider`（显式启用 TensorRT 时也可能为 `TensorrtExecutionProvider`）。

如果诊断明确提示模型损坏，可强制重新下载后再诊断：

```powershell
.\.venv\Scripts\python.exe -m face_matching.model_manager --force
```

## 鼠标操作

1. 打开“人员库”，点击“录入人员”。填写姓名、身份证号，选择一张或多张仅包含该人员的人脸照片。
2. 建议每人录入 3～8 张清晰照片，覆盖正面、左右轻微侧脸、眼镜和实际摄像头光照。不要录入严重模糊、多人合照或极端大角度照片。“照片管理”可删除重复或低质量样本，但每人至少保留一张。
3. 返回“实时识别”，选择“打开视频”“摄像头”或“RTSP / 网络流”，点击“开始识别”；可用鼠标暂停/继续。
4. 相似度阈值默认 `0.50`。阈值越高误报越少、漏报越多；部署前必须用目标摄像头采集的正负样本校准，不能把界面分数当作概率。
5. “处理帧步长”默认 `1`，效果优先。显卡负载过高时可调大以提升速度，但快速经过的人脸可能漏检。

可选命令行视频源仅用于调试：

```powershell
.\.venv\Scripts\python.exe -m face_matching.app --source "rtsp://user:password@camera/stream"
```

## 数据与配置

默认数据位于 `%LOCALAPPDATA%\FaceMatching`：

- `face_matching.sqlite3`：人员信息、特征向量、照片索引；
- `photos\`：录入照片的本地副本；
- `models\`：源码安装下载的 ONNX 模型；离线发布包优先读取包内只读模型。

可用环境变量：

| 变量 | 用途 |
| --- | --- |
| `FACE_MATCHING_HOME` | 整体数据目录 |
| `FACE_MATCHING_MODEL_DIR` | 模型目录 |
| `FACE_MATCHING_DATABASE` | SQLite 数据库路径 |
| `FACE_MATCHING_DETECTOR_MODEL` | 自有 SCRFD 兼容 ONNX 模型 |
| `FACE_MATCHING_RECOGNIZER_MODEL` | 自有 LVFace 兼容 ONNX 模型 |
| `FACE_MATCHING_MODEL_ID` | 特征版本；更换识别模型时必须同时修改 |
| `FACE_MATCHING_TENSORRT=1` | 已正确安装 TensorRT 时优先使用它，否则默认 CUDA |

模型或特征预处理改变后旧特征不能混用，应使用新的 `FACE_MATCHING_MODEL_ID` 并重新录入照片。v0.2.2 默认 ID 是 `lvface-b-glint360k-v2-tta`，因为加入镜像 TTA，旧版人员库需要重新录入照片。

## 准确率和使用边界

- 小于约 36 像素、严重运动模糊、遮挡严重或接近纯侧脸的人脸，本身缺少足够身份信息；系统会优先显示“质量不足/未知”。
- 默认阈值只是起点。正式部署应按摄像头、距离和人群做验证集，基于可接受的 FAR（误认率）选阈值。
- 当前实现是人脸检索，不包含活体检测，不能单独用于门禁放行、支付、执法定案等高风险自动决策。
- 身份证号和生物特征属于敏感个人信息。本版在本机 SQLite 和照片目录中存储，未做磁盘加密；生产环境需增加访问控制、磁盘/数据库加密、审计、保留期和合法授权。

## 模型许可证

项目不会把模型权重提交到 Git。安装脚本下载的 InsightFace/SCRFD 与 LVFace 预训练权重仅适用于**非商业研究**；LVFace 代码本身是 MIT，但其官方 README 对下载权重另有限制。商业或生产使用必须替换为获得相应授权的 ONNX 权重，并重新完成阈值和偏差评估。

## 开发验证

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall -q src tests
```

单元测试不需要 GPU；应用启动和 `face-matching-diagnose` 的完整自检始终需要 GPU。
