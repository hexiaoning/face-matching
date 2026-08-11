# 监控视频人脸比对

面向 Windows 11 + NVIDIA GPU 的本地桌面程序。人员库保存“姓名 + 身份证号 + 1～多张照片”，可用鼠标完成录入、编辑、删除，以及视频文件、本地摄像头和 RTSP 监控流识别。

当前版本已实现完整可运行链路；AI 推理强制使用 CUDA，GPU 或 CUDA 执行器不可用时直接报错退出，不会静默降级到 CPU。

## 方案选择

监控视频的难点不是单张高清正脸识别，而是小脸、运动模糊、侧脸、遮挡和开放集误认。实用的 SOTA 方案应是一条管线，而不是只替换一个网络：

```text
视频 / 摄像头 / RTSP
        ↓
10GF 人脸检测 + 5 点关键点
        ↓
相似变换对齐 + 清晰度 / 姿态 / 尺寸 / 光照质量评估
        ↓
LVFace-B 512 维身份特征（CUDA）
        ↓
轨迹关联 → 保留优质帧 → 异常帧剔除 → 质量加权聚合
        ↓
人员多照片模板 1:N 检索
        ↓
绝对阈值 + 第一/第二候选间隔 → 已知人员或未知人员
```

选择依据：

- 识别采用 [LVFace（ICCV 2025 Highlight）](https://openaccess.thecvf.com/content/ICCV2025/html/You_LVFace_Progressive_Cluster_Optimization_for_Large_Vision_Models_in_Face_Recognition_ICCV_2025_paper.html) 的 B 型官方 ONNX 权重。它是目前有公开推理代码和权重、又便于 Windows GPU 部署的最新高精度方案之一；论文报告其在 IJB-C 等基准超过此前方法，并曾取得 MFR-Ongoing 学术榜第一。
- 检测采用 [InsightFace 10GF 检测器](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)，输出人脸框和五点关键点。大模型检测器比移动端版本更适合监控画面中的小脸。
- 低质量专项研究中，[AdaFace](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html) 和 [ARoFace（ECCV 2024）](https://arxiv.org/abs/2407.14972) 在 IJB-S/TinyFace 上很有竞争力。但其最强公开权重没有官方部署级 ONNX 流程，因此本版没有把研究训练仓库直接拼进生产依赖。
- 视频不是逐帧独立投票。本版按轨迹做质量加权特征聚合，这与视频人脸识别中质量感知聚合的结论一致，例如 [C-FAN](https://arxiv.org/abs/1902.07327)。
- 默认不对人脸做生成式超分辨率或修复。严重模糊里已经丢失的身份信息无法可靠恢复，生成模型可能“补出”另一张脸；本版会等待同一轨迹中的更好帧，始终不足则显示低质量/未知。

这仍不意味着任何模糊或纯侧脸都能识别。若脸部只有十几个像素、完全背对镜头或整段都严重拖影，可靠系统应拒绝判断，而不是给出一个看似确定的人名。

## 已实现功能

- 图形化人员库：新增、编辑、删除和预览多张照片。
- 中文路径支持；录入照片必须只含一张脸，低质量照片会给出明确错误。
- 视频文件、本地 USB 摄像头、RTSP/RTSPS/HTTP 网络流。
- 多人同时检测、轨迹编号、优质帧积累和识别事件列表。
- 身份证号唯一约束；列表和识别事件默认脱敏显示。
- SQLite 本地数据库，照片和特征均不上传。
- 模型下载 SHA-256 校验和版本隔离。
- ONNX Runtime CUDA 会话禁用 CPU EP 回退；模型真实预热失败则拒绝启动。
- CUDA 12 / cuDNN 9 运行库随 Python 依赖安装，无需单独安装完整 CUDA Toolkit。

## Windows 11 一键运行

目标配置：RTX 4070 SUPER 12GB、64GB 内存、Windows 11 64 位。其他支持 CUDA 12 的 NVIDIA GPU 也可以使用；建议至少 8GB 显存。

1. 安装或更新 NVIDIA 显卡驱动。无需安装 CUDA Toolkit。
2. 双击 `install_and_run.bat`。
3. 首次启动阅读并确认模型许可，程序会下载约 710 MiB 模型并校验完整性。
4. 后续双击 `run.bat` 即可启动。

脚本会创建隔离的 `.venv`，安装 Python 包以及 CUDA/cuDNN 运行时。若电脑没有 Python，它会优先通过 Windows `winget` 为当前用户安装 Python 3.12；没有 `winget` 时会提示手动安装 64 位 Python 3.11～3.13。

命令行安装方式：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m face_match
```

依赖方案依据 ONNX Runtime 官方的 [CUDA Execution Provider 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)：`onnxruntime-gpu[cuda,cudnn]` 会安装运行 DLL，应用启动时调用 `preload_dlls()` 从 Python 包加载这些 DLL。

## 鼠标操作

1. 打开“人员库”，点击“新增人员”。填写姓名和身份证号，添加一张或多张照片后保存。
2. 建议每人录入至少三张清晰照片：正面、左侧约 30～60°、右侧约 30～60°。录入照与摄像头年龄、光线和拍摄高度越接近越好。
3. 打开“实时识别”，选择“打开视频”“本地摄像头”或“RTSP 网络流”。
4. 绿色框表示通过双重阈值，橙色表示仍在收集优质帧，红色表示未知人员。
5. 右侧事件表显示命中的姓名、脱敏身份证号、余弦相似度和画面质量。

RTSP 地址中的账号密码只用于当前连接，不写入设置或数据库。示例：

```text
rtsp://username:password@192.168.1.20:554/stream1
```

## 阈值不是“置信概率”

界面默认相似度阈值为 `0.48`，并要求第一名至少领先第二名 `0.04`。显示的分数是经过多模板融合的余弦相似度，不是“正确率 48%”。默认值只能作为起点，正式部署必须用现场摄像头数据标定：

1. 收集同人和不同人的实际视频样本，覆盖白天/夜间、距离、侧脸、眼镜和运动模糊。
2. 先根据可接受的误报率选阈值，再检查漏报率；安防场景通常应优先压低误报。
3. 对不同摄像头分别验证。镜头高度、码率或补光改变后重新验证。
4. 不要把自动命中作为处罚、执法或身份最终结论；高风险用途必须人工复核。

## 数据位置与备份

默认数据目录：

```text
%LOCALAPPDATA%\SurveillanceFaceMatch\
├── faces.sqlite3       # 姓名、身份证号、特征和照片索引
├── photos\             # 录入照片的本地副本
├── models\             # 校验过的 ONNX 模型
└── settings.json
```

可以在启动前设置 `FACE_MATCH_DATA_DIR` 改变位置。身份证号和生物特征属于敏感信息；建议使用 BitLocker、限制 Windows 目录权限、制定删除期限，并只允许有合法授权的操作员访问。备份时应整体备份上述目录，不要只复制 SQLite 文件。

## GPU 故障处理

程序不会使用 CPU 做神经网络推理。常见错误：

- `未检测到 CUDAExecutionProvider`：通常安装了错误的 `onnxruntime` CPU 包。重新运行 `install_and_run.bat`。
- `CUDA/cuDNN 运行库加载失败`：删除 `.venv` 后重新运行安装脚本，或检查安全软件是否隔离了 DLL。
- `模型无法在 CUDA GPU 上完整加载`：更新 NVIDIA 驱动；RTX 40 系列建议使用当前稳定驱动。
- 显存不足：关闭占用显存的游戏、生成式 AI 或视频程序后重试。

视频解码、图像缩放、数据库和界面仍会使用少量 CPU；“GPU-only”指检测与识别神经网络严禁落到 CPU。

## 开发与测试

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m compileall -q face_match tests
pytest
```

核心单元测试不需要 GPU；实际模型会话和端到端视频验收必须在 NVIDIA Windows 目标机上运行。

## 许可证

本项目代码为 MIT License。自动下载的预训练模型不是同一许可范围：LVFace 官方仓库和 InsightFace 均声明其发布权重仅限非商业研究用途。商业使用前应联系相应权利方取得许可，不能仅凭本项目的 MIT License 推定模型可商用：

- [LVFace 官方仓库许可说明](https://github.com/bytedance/LVFace#license)
- [InsightFace 官方许可说明](https://github.com/deepinsight/insightface#license)
