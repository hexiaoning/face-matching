# Face Matching

Windows 11 桌面端图片 / 视频人脸检索。人员库保存“姓名 + 身份证号 + 1～多张照片”，界面支持本地图片、视频、摄像头、RTSP/RTMP 视频流，识别结果会在人脸框和匹配列表中显示。每个命中人员保留最高 score，并在匹配列表中按 score 倒序输出；本地视频文件会按机器性能尽快处理，界面预览限制为 15 FPS，避免高速处理时积压图像。支持“不是此人”纠错、右键关联已录入人员，以及导出标注库。

> 这是 **GPU-only** 应用。NVIDIA 使用 ONNX Runtime CUDA，Intel 核显/Arc 使用 OpenVINO GPU FP32；两种后端都禁止神经网络回退 CPU。`auto` 优先 CUDA，再选 OpenVINO。视频解码、跟踪管理、SQLite 和 GUI 等非神经网络工作仍由 CPU 执行。

CUDA 启动时执行真实推理并检查节点执行记录，要求存在 CUDA 计算节点；只允许少量整数/布尔张量的形状辅助操作在 CPU 上运行，卷积、矩阵乘法和浮点特征计算落到 CPU 时仍会拒绝启动。这兼容 ONNX Runtime 默认的形状节点调度，避免 SCRFD 在正常 CUDA 环境中因 `session.disable_cpu_ep_fallback` 而无法加载。

> 这是部署在受控环境中的**内部系统，不需要脱敏功能**。程序不遮罩身份证号，也不清理视频源地址中的用户名或密码；人员表、识别结果和事件日志均保留并显示完整原始值。部署方应通过内部账号权限、主机访问控制和磁盘加密保护这些数据。

## 方案

```text
视频/摄像头
   → SCRFD-10G 人脸检测 + 5 点对齐 (CUDA / OpenVINO GPU)
   → 模糊/姿态/尺寸/照明质量门控
   → LVFace-B 特征 + 水平镜像测试增强 (CUDA / OpenVINO GPU)
   → 位置 + 运动预测 + 外观特征的人脸轨迹关联
   → 同一轨迹 Top-K 高质量帧加权聚合 + 离群身份帧剔除
   → 每人多照片模板（最佳样本 + Top-3 质量加权 + 人员中心）
   → 开集判定（相似度阈值 + 第一/第二候选差值 + 多帧共识）
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

构建脚本会先跑测试，下载并校验固定 SHA-256 的模型，然后把 Python、PySide6、OpenCV、CUDA/OpenVINO GPU 运行库和两套模型一并打入 `dist\FaceMatching-v2.6.3-windows-x64.zip`。目标机不需要 Python、CUDA Toolkit、OpenVINO Toolkit 或互联网；只需预装对应显卡驱动。

目标机可直接双击 `FaceMatching-v2.6.3-Setup.exe` 安装；也可解压便携 ZIP 后双击 `安装并启动 Face Matching.cmd`。启动窗口会先显示 `Face Matching Offline Bundle v2.6.3` 和构建标记，再离线校验 Qt GUI 运行库、GPU 和模型。两种方式都不会调用 `winget` 或 `pip`。

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

1. 打开“人员库”，点“录入人员”，填姓名和身份证号，选择 1～多张照片。建议同时提供清晰正面、左右轻度侧脸照片。选中人员后点“查看照片…”，或双击人员行，可查看已录入照片的缩略图、大图，并切换多张照片。
2. 在“人脸检索”中一次选择多张图片、多个视频（可混选），选择后自动排队比对，无需点击“开始识别”。处理中可继续选择文件添加到队列，点击“停止”取消剩余队列。也可直接输入文件路径（多个本地路径用分号分隔）、`0` 或 RTSP/RTMP 地址，输入完成后自动开始；“摄像头 0”按钮直接启动。图片支持 JPG、PNG、BMP、WebP、TIFF 等格式，可包含多个人；粘贴 `data:image/...` 图片数据时会先解码保存到本地，再执行单张图片检索，来源和标注可正常导出。
3. 图片逐脸单次匹配，不等待视频多帧确认；质量、相似度和候选差值门槛仍有效。绿色为已确认人员，黄色为采集中/待确认。未通过相似度或候选差值门槛时，右侧仍会展示最佳相似候选，标为“待确认”；悬停可查看分数和未通过的原因。待确认不计入确认命中人数、不写入确认识别事件，需人工核实并右键关联。自动确认和人工关联优先于待确认候选，同类结果在每个来源中每人保留最高 score。批量结果累计显示来源列，轨迹和标注独立；人数按人员去重。单个来源处理失败后继续剩余队列，最后提示失败项。
4. 右侧每行都有“是他”和“不是他”两个标注按钮。点“是他”可人工确认该候选，点“不是他”排除该轨迹与该人员的匹配；已选按钮会高亮，再次点击可撤销。切换选择会在同一事务中取消相反标注并保存新标注，保留完整历史。右键点击左侧人脸框，选择“关联已录入人员…”，搜索姓名或身份证号并选择人员。人工确认、人工关联均显示蓝框，优先于自动匹配，保留原始评分；没有自动评分的关联显示“—”。标注只作用于本次检索的对应轨迹，不会自动新增底库照片或修改模型。
5. 在“标注库”查看持久保存的标注和操作历史，以及“有效 / 历史记录”状态。双击标注行或点“查看样本对比…”，并排查看标注原图 / 视频帧、人脸截图和对应人员库照片，可切换多张人像。缺失的标注帧优先从本地来源文件恢复显示。支持 Ctrl / Shift 多选；有选中记录时只导出选中记录及其媒体、人员，点“取消选择”后导出全部。视频默认导出标注位置前后各 5 秒的片段（可调整，片段无音轨），也可选择原样导出完整视频。来源媒体、人员人像、人员其它信息可独立勾选。
6. 导出 ZIP 包含 `annotations.json`（导出范围内的操作历史）、`current_annotations.json`（范围内仍有效的标注）、`manifest.json`（范围、选项和缺失清单），以及勾选的 `media/`、`annotations/` 标注画面、`people.json` 和 `people/` 录入人像。有效状态始终依据完整历史判断，再取导出范围；只选中已撤销或替换的旧标注不会恢复其有效性。每条记录的 `is_current` 表示是否有效，`superseded_by` 指向后续操作 ID（可能未选中、不在本次导出内）。标注记录保留来源、会话、轨迹、原始帧号、人脸框、人员、操作、时间、模型和原始评分；视频片段另外记录起始帧、片段内标注帧和帧率。摄像头/直播流目前不保存完整录像，导出可用的标注画面或人脸截图；来源文件被移走时会提示缺失。
7. 在“设置”中调 GPU 后端、相似度、质量、检测尺寸等参数。更改识别模型或镜像 TTA 后需重启，并在人员库中点“重建当前模型特征”。

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
- `data/annotations/`：标注时保存的画面和人脸截图（标注记录在 SQLite 中）
- `data/query_images/`：从粘贴图片数据保存的检索原图，供后续标注和导出使用
- `config.json`：界面设置

可用 `FACE_MATCHING_HOME` 环境变量改变数据目录。身份证号和人脸特征属于敏感个人信息；生产部署应使用 BitLocker/EFS 或组织的密钥系统加密数据目录，并设置访问权限、保留期和审计。

## 开发验证

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall -q src
```

纯逻辑测试可在无 GPU 的开发机上执行；真实模型启动必须通过 CUDA 或 OpenVINO GPU 硬校验。
