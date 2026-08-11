# Face Matching

面向监控视频的本地人脸比对桌面应用。人员底库由“姓名 + 身份证号 + 1～多张照片”组成，支持视频文件、本机摄像头及 RTSP/RTMP/HTTP 视频流；所有操作均可通过图形界面和鼠标完成。

本实现**强制使用 NVIDIA GPU**。CUDA 推理不可用时会明确报错并退出，不会静默切换到 CPU。

## 首版已实现

- PySide6 中文桌面 GUI：实时识别、人员库管理、参数设置及运行状态。
- SQLite 本地人员库；一个人员可录入多张正脸/侧脸照片，身份证号唯一。
- 视频文件、USB 摄像头、RTSP/RTMP/HTTP 监控流。
- SCRFD-10G 多尺度人脸检测和 5 点对齐。
- AntelopeV2 R100 人脸特征，按人员融合多个照片模板。
- 监控轨迹级识别：IoU 跟踪、模糊/尺寸/姿态/曝光质量评分、质量加权多帧特征、连续命中确认。
- 识别框、相似度、质量、脱敏身份证号以及最近确认记录。
- ONNX Runtime 只注册 `CUDAExecutionProvider`，并关闭 CPU EP fallback。
- 首次启动自动下载官方模型，使用 SHA-256 校验；照片和模型下载均支持中文路径。
- Windows 一键安装；pip 包自带 CUDA/cuDNN 运行库，目标机器无需安装 CUDA Toolkit。

## “SOTA”方案怎么选

低清、运动模糊、非正脸、遮挡和开放集误报是不同问题，实际最强方案不是换一个模型就结束，而是一条完整链路：

1. **大输入多尺度检测及可靠关键点**：SCRFD/RetinaFace；对 1080p 监控画面使用 960 或 1280 检测输入，避免小脸在缩放时消失。
2. **低质量/姿态鲁棒特征**：研究路线优先考虑 [CVLFace 的 ViT-KP-RPE + AdaFace](https://github.com/mk-minchul/CVLface)。官方表中 WebFace12M 权重在 TinyFace Rank-1 为 76.10，AdaFace 本身就是针对低质量人脸设计的。
3. **视频集合聚合**：不要逐帧独立决定身份。使用跟踪后的人脸集合做质量感知融合；进一步升级可用 [CAFace](https://github.com/mk-minchul/caface)。
4. **识别效用质量模型**：生产版可把当前轻量启发式质量分替换为 CR-FIQA、MR-FIQA 或 OFIQ，并丢弃无识别价值的帧。
5. **开放集标定**：不能照搬网上的阈值。必须用实际摄像头、距离、光照及人员分布采集正负样本，按可接受的 FAR（误识率）确定阈值。

本首版默认采用 [InsightFace AntelopeV2](https://github.com/deepinsight/insightface/tree/master/model_zoo)（SCRFD-10G + R100@Glint360K）。它不是最新论文排行榜上每一项都最强的组合，但 ONNX 部署成熟、RTX 4070 SUPER 性能充足、Windows 安装链短，适合作为可运行基线。工程上增加的轨迹、多帧质量融合和确认机制，通常比把模糊单帧直接送进另一个静态模型更重要。

不会把 GFPGAN、扩散模型等生成式“人脸修复”后的图片用于身份特征：这类模型可能生成原视频中不存在的身份细节。它们最多只能用于另行标注的人工查看画面。

## Windows 11 安装

目标设备：华硕 ROG G22CH、i7-14700KF、RTX 4070 SUPER 12 GB、64 GB 内存、Windows 11 64 位。

### 最简单方式

1. 安装较新的 NVIDIA 显卡驱动。**不用安装 CUDA Toolkit。**
2. 双击 `install.bat`。脚本会：
   - 检查 NVIDIA GPU/驱动；
   - 在缺少 Python 时通过 `winget` 安装 64 位 Python 3.12；
   - 创建隔离的 `.venv`；
   - 安装 GUI、OpenCV、ONNX Runtime GPU 以及随 pip 分发的 CUDA/cuDNN DLL；
   - 验证 `CUDAExecutionProvider`。
3. 双击 `run.bat`。
4. 首次启动确认模型的非商业研究许可后，程序下载约 344 MiB 的 AntelopeV2 模型并校验 SHA-256。

CUDA/cuDNN 由 Python 依赖提供这一做法来自 [ONNX Runtime CUDA Execution Provider 官方说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)：新版本可以预加载 PyTorch 或 NVIDIA site-packages 中的运行库，因此不要求系统级 CUDA Toolkit。

### 命令行方式

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[gpu]"
python -m face_matching --check-gpu
python -m face_matching
```

指定数据目录：

```powershell
python -m face_matching --data-dir D:\FaceMatchingData
```

## 使用

### 录入人员

进入“人员库”→“录入人员”，填写姓名和身份证号，选择一张或多张照片。每张照片必须只包含一个人；系统会在保存前用 GPU 检测、对齐并提取特征。

建议每人录入 3～5 张清晰照片：一张正脸、左右各一张轻度侧脸，尽量覆盖监控摄像头的拍摄角度。底库照片质量远比盲目增加数量重要。小于 40 像素或质量过低的人脸会被拒绝。

### 开始识别

在“实时识别”中选择：

- “打开视频”：MP4、AVI、MKV、MOV 等；
- “连接摄像头”：输入摄像头编号，通常为 `0`；
- “连接 RTSP”：粘贴完整视频流地址。

黄色框代表未识别或正在确认，绿色框代表达到连续命中次数的已确认人员，红色框代表当前帧质量不足。RTSP 用户名和密码不会写入识别事件记录或显示在状态栏。

### 参数

- **相似度阈值**：默认 0.50；调高会降低误报、增加漏报。
- **最低帧质量**：低于此值不提取/累积身份特征。
- **检测分辨率**：小脸较多时选 960 或 1280；更高更准也更慢、更占显存。
- **每 N 帧提取特征**：检测仍逐帧执行，身份特征按间隔批量执行。
- **连续命中次数**：同一轨迹连续多少次得到同一身份后才确认。

修改参数后会在下一次打开视频源时完整生效。

## 数据与隐私

默认数据位于项目目录下的 `data/`：

```text
data/
├── face_matching.sqlite3   # 人员、特征及识别事件
├── gallery/                # 录入照片的本地副本
├── models/antelopev2/      # SHA-256 校验后的 ONNX 模型
└── settings.json
```

身份证号和人脸模板属于高度敏感的生物识别/身份数据。当前版本是单机原型，未提供磁盘加密、账号权限、审计导出或联网同步。正式部署至少应使用 BitLocker/加密卷、操作员鉴权、最小权限、数据保留期限和访问审计，并确认采集及识别符合所在地法律和授权范围。

## 模型许可

- 本仓库代码使用 MIT 许可。
- InsightFace 的**代码**为 MIT 许可，但其官方提供的预训练模型（包括自动/手动下载的 AntelopeV2/Buffalo 系列）声明为**仅限非商业研究**。商业部署应按 [InsightFace 官方许可说明](https://github.com/deepinsight/insightface#license) 联系模型方取得授权，或替换成自行训练且数据来源合规的 ONNX 模型。
- 代码没有把模型二进制提交进仓库；首次运行由操作者确认后从官方 GitHub Release 下载。

## 开发与测试

不含 GPU 的机器仍可运行纯逻辑测试，但应用本身会按设计拒绝启动：

```bash
python -m pip install -e ".[gpu,dev]"
pytest
python -m compileall -q src tests
```

GPU 机器上的验收：

```powershell
python -m face_matching --check-gpu
```

然后至少验证：多人同框、正侧脸转换、运动模糊、遮挡、夜间画面、陌生人开放集、视频结束、RTSP 断线，以及录入/编辑/删除后底库即时更新。

## 当前边界

- 极小脸（大约低于 30～40 像素）、严重运动模糊或完全侧脸没有可靠身份信息，任何模型都无法保证恢复。
- 当前跟踪器是轻量 IoU 跟踪，拥挤交叉场景可升级为 ByteTrack/BoT-SORT，并用人脸特征辅助重关联。
- 当前质量分是无需额外模型的清晰度、尺寸、姿态、曝光组合；建议生产版接入 CR-FIQA/MR-FIQA 并在现场数据上验证。
- 当前未实现活体检测。用于门禁或身份核验时，应增加近红外/深度或经过攻击测试的活体方案，不能只靠 RGB 人脸匹配。
