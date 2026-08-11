# Face Matching

Windows 11 桌面人脸比对工具：从视频文件、本机摄像头或 RTSP 监控流检测人脸，和本地人员库做 1:N 开集比对。人员资料为“姓名 + 身份证号 + 1～多张注册照”。界面全部可用鼠标操作。

> 识别结果是概率判断，不是身份证明。模糊、遮挡、极端侧脸和图库规模扩大都会增加误识风险，任何实际处置都必须人工复核，并遵守所在地关于生物特征、监控和个人信息处理的法律。

## 本版实现

- **GPU-only**：ONNX Runtime CUDA，显式设置 `session.disable_cpu_ep_fallback=1`。没有 NVIDIA CUDA 执行器时直接报错，不会偷偷用 CPU。
- **检测与对齐**：SCRFD-10G 检测正面和非正面人脸，使用 5 点关键点进行相似变换对齐。
- **识别**：默认 ArcFace ResNet-50 ONNX；原图和水平翻转图做批量 TTA 后融合特征；同一人保留每张注册照的独立模板，按身份取最佳模板得分。
- **低清视频处理**：默认用 960×544 检测 16:9 监控画面；按检测置信度、脸部尺寸、清晰度和姿态估计计算画质权重；同一轨迹跨帧质量加权聚合，避免由某一个模糊帧决定结果。
- **开集判定**：同时要求最佳相似度超过阈值，且与第二候选保持最小差值；否则显示“待人工复核/未知”。
- **稳定确认**：画质低于下限的轨迹不会自动确认；默认同一候选需连续 3 个推理帧通过阈值和 Top-2 差值后才显示“已匹配”。
- **隐私**：身份证号使用 AES-256-GCM 加密；Windows 下主密钥由 DPAPI 绑定当前登录用户。界面只显示脱敏值，视频和日志不出现身份证号。
- **数据本地化**：SQLite、注册照、模型和配置默认位于 `%LOCALAPPDATA%\FaceMatching\FaceMatching`。

## 为什么这样选

低质量监控人脸识别没有一个模型能单独解决全部问题，实用方案通常是：强检测器 → 关键点对齐 → 质量鲁棒的特征模型 → 多模板/多帧聚合 → 经过现场数据校准的开集阈值。

调研结论：

1. [AdaFace（CVPR 2022）](https://github.com/mk-minchul/AdaFace)针对低质量人脸调整分类间隔，在 IJB-S、TinyFace 等低清/监控基准上优于同训练集的 ArcFace；其官方仓库也明确展示了模糊视频优势。
2. [CVLFace / KP-RPE](https://github.com/mk-minchul/CVLface)把关键点相对位置编码引入 ViT，对对齐误差、非正面和 TinyFace 更有优势。其官方性能表中，ViT KP-RPE + AdaFace + WebFace12M 的 TinyFace Rank-1 为 76.10，是本项目调研到的最适合该场景的公开研究方案。
3. [CAFace](https://github.com/mk-minchul/CAFace)面向长视频模板聚合；在人员停留时间较长、轨迹较完整的离线视频分析中，比简单平均更有潜力。
4. 工程首版选择 [InsightFace](https://github.com/deepinsight/insightface) 的 SCRFD + ArcFace ONNX，是因为 Windows/CUDA 部署成熟、RTX 4070 SUPER 上有充足余量，且易于替换识别模型。官方模型表中 `buffalo_l` 使用 SCRFD-10GF 和 ResNet50@WebFace600K。
5. CUDA Toolkit 无需单独安装：[ONNX Runtime 官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)支持通过 `onnxruntime-gpu[cuda,cudnn]` 安装运行时 DLL，并用 `preload_dlls()` 加载。

本仓库默认可下载的 `buffalo_l` 是一个可运行工程基线，并不代表 2026 年绝对最高精度。真正的高精度生产方案应将识别器替换为经过授权并在自有摄像头数据上验证的 AdaFace IR-101 / CVLFace 模型，再做阈值校准。KP-RPE 需要关键点作为模型额外输入，不能直接当作单输入 ArcFace ONNX 替换。

## 目标环境

| 项目 | 配置 |
| --- | --- |
| 系统 | Windows 11 64 位 |
| GPU | NVIDIA RTX 4070 SUPER 12 GB（必须） |
| CPU / 内存 | Intel i7-14700KF / 64 GB |
| 驱动 | 近期 NVIDIA 官方驱动 |
| CUDA Toolkit | 不需要单独安装 |

开发环境使用 Python 3.11 或 3.12；安装器会自动准备 Python 3.12。

## 安装和启动

1. 安装最新 NVIDIA 官方显卡驱动。
2. 双击 `install.bat`。脚本会检查 NVIDIA GPU、安装 `uv`、Python 3.12、GUI、ONNX Runtime GPU 及 CUDA/cuDNN 运行库，并验证 CUDA 执行器。
3. 双击 `run.bat`。
4. 首次启动：
   - 非商业研究：在弹窗中确认许可证后，用鼠标下载约 275 MiB 的 `buffalo_l`；下载完成会校验 SHA-256。
   - 商业/生产：在“设置与模型”中选择持有合法授权的模型目录。
5. 在“人员库”录入人员；每张注册照只能有一张合格人脸，低于画质下限的照片会被跳过。再到“视频识别”选择视频、摄像头或 RTSP 地址。

所有模型推理必须落在 ONNX Runtime CUDA Execution Provider。OpenCV 只负责视频解码、缩放、对齐和画面绘制；如果 CUDA Provider 或 CUDA/cuDNN DLL 不可用，应用会在模型初始化时明确报错并停止，不会退回 CPU 跑神经网络。

开发运行：

```powershell
uv sync --extra dev
uv run face-matching
uv run pytest
uv run ruff check .
```

## 自有 ONNX 模型

模型目录支持以下两套文件名：

- 默认：`det_10g.onnx` + `w600k_r50.onnx`
- 自有：`detector.onnx` + `recognizer.onnx`

检测器必须兼容 SCRFD 的 5 点关键点输出。识别器必须接收 `N×3×112×112` 并输出 `N×D` 嵌入。默认预处理是 RGB、`(x - 127.5) / 127.5`。AdaFace 官方权重通常使用 BGR；在同目录放置：

```json
{
  "recognizer_color_order": "bgr",
  "recognizer_mean": 127.5,
  "recognizer_std": 127.5
}
```

更换识别器后，已有向量不会混用（模型哈希及预处理参数参与 `model_id`）；需要用新模型重新录入人员。

GUI 可关闭水平翻转 TTA 以降低识别网络计算量；开关状态参与 `model_id`，注册和视频探针始终使用相同设置。切换后需要按新设置重新录入人员。

KP-RPE 等含额外关键点输入的多输入模型目前需要另写适配器，不能直接改名使用。

## 阈值与现场验收

默认相似度阈值 `0.45` 只用于启动验证，不是生产保证。推荐收集目标摄像头的正样本和足够多的非同人样本，分别按白天/夜间、距离、俯仰角、运动模糊分层评估：

1. 先确定可接受的误报率（FAR），再从非同人得分分布选择阈值。
2. 在该阈值报告漏报率（FRR）、Rank-1、未知人员拒识率，并按人群和场景检查差异。
3. 图库人数增大后重新校准；不能把 LFW 等网页照片准确率当作监控现场准确率。
4. 将“最小人脸边长”设为现场可验证的下限。低于约 30～40 像素的人脸通常不应自动确认。
5. 超分辨率可能改善观感，却不能恢复已经丢失的身份信息；不得把生成细节当作可靠证据。

建议从默认 `960×544 / 每 2 帧推理 / 连续 3 次确认 / 最低画质 0.35` 起步。若镜头中人脸很小，可把检测输入提高到 `1280×728`；若帧率不足，优先增加抽帧间隔，不要关闭 CUDA 或降低误报控制条件。

## 许可证

- 本项目代码：MIT，见 `LICENSE`。
- InsightFace 代码为 MIT，但其官方训练数据及预训练模型（包括自动/手动下载的 `buffalo_l`）按官方说明仅限非商业研究；2025-11 的官方说明要求开源识别权重另行联系授权。应用不会静默下载，必须由用户确认。
- 商业部署应联系模型权利方获得授权，或使用自训/已授权权重，并确认训练数据、模型权重及生物特征处理均合规。

## 当前边界

- 本开发机没有可用的 `nvidia-smi`，因此代码可以验证“无 GPU 明确失败”和纯逻辑测试，但最终 CUDA 吞吐、显存占用及摄像头协议必须在目标 RTX 4070 SUPER 上验收。
- 当前为单视频源桌面版；多路摄像头、跨镜追踪、告警联动、活体检测、审计权限和高可用服务不在首版范围。
- 未保存识别事件或视频截图，避免不必要的敏感数据留存。
