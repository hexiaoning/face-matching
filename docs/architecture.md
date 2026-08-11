# 方案选型与边界

## 为什么是这套方案

“监控人脸识别 SOTA”不是单一模型名称，而是检测、对齐、低质量特征、视频聚合和开集校准的系统问题。此实现可在 Windows + NVIDIA CUDA 或 Intel OpenVINO GPU 上交付。

### 检测：SCRFD-10G

SCRFD 是 ICLR 2022 的高效人脸检测方案，对小人脸的精度/算力平衡好，同时输出 5 个对齐关键点。此项目使用 10G 版本，困难监控默认 960×960；显卡压力过大可改为 640，远距离人脸可改为 1280。

### 识别：LVFace-B 与 AuraFace

- **LVFace-B**：ICCV 2025 Highlight，官方 Glint360K 权重的 IJB-C TAR 在 FAR=1e-6/1e-5/1e-4 下为 90.06/97.00/97.70。官方提供 ONNX，但权重仅限非商业研究。
- **AdaFace / KP-RPE**：AdaFace 的训练目标明确面向低质量人脸，官方在 IJB-S/TinyFace 上报告了强结果；KP-RPE 进一步增强姿态/对齐鲁棒性。它们是重要的研究对比，但官方发布以 PyTorch 训练/评测链为主，不是本项目需要的直接、可校验 ONNX Windows 交付。
- **AuraFace-R100**：准确率不作为 LVFace-B 的等价 SOTA 声明；它是 Apache-2.0 的工程备选，用来解决 LVFace 权重的非商业限制。

因此此版默认用 LVFace-B 追求精度，并提供 AuraFace 可切换档位。如果项目获得可商用的 AdaFace/KP-RPE ONNX 权重，现有 `ModelProfile` 和 `FaceEmbedder` 边界可直接扩展。

### 视频：不依赖单帧

监控视频中并非每一帧都同样可用。本实现会：

1. 根据拉普拉斯清晰度、5 点几何姿态、人脸尺寸、检测置信度和照明评分，用加权几何平均防止严重模糊/侧脸被其他高分掩盖。
2. 对达到最低质量的帧做原图 + 水平翻转特征平均；低姿态质量或低检测置信度帧额外尝试一个确定性的轻微对齐偏移。
3. 按位置重叠、速度预测与外观特征关联轨迹。一个视频帧即使产生多个对齐特征，也只计为一份独立支持证据。
4. 每帧对目标照片的多个参考模板取最佳相似度；至少多个独立帧达到确认阈值才自动命中。单帧弱命中只进入“待复核”，不会伪装成已确认。
5. 本地文件记录媒体时间而非处理机器的当前时间，并保留候选人脸缩略图、综合分、最佳帧分和支持帧数。

这是面向交付的质量加权聚合，不声称完整复刻 CAFace 等需要专用训练的集合模型。

6. 镜像 TTA 改变特征空间，因此 `model_id` 显式包含 `-tta` 或 `-single`，切换后要求重建底库特征。

### 目标照片：紧裁近景与对齐误差

目标照片常常是自拍或已裁好的人脸，脸占满画面时反而超出检测器训练中常见的尺度。参考照片检测会在周围添加中性灰色画布并以多个上下文尺度重新检测，再把框和关键点严格映射回原图。此过程不做人脸生成或超分辨率，也不会补造五官细节。目标模板使用三个小幅水平对齐偏移，提高五点关键点在俯拍、侧脸下的容错。

目标检索属于一对一验证，不使用人员库一对多识别的第一/第二名 margin。它可以使用更低的候选阈值，但必须同时满足独立帧支持数；默认确认线只是针对当前模型的工程起点，部署前仍需按现场 FAR/TAR 校准。

## 硬性 GPU 约束

- NVIDIA 路径验证 `onnxruntime-gpu`、`CUDAExecutionProvider` 和 `nvcuda.dll`，并设置 `session.disable_cpu_ep_fallback=1`。
- Intel 路径用 OpenVINO 直接把 ONNX 编译到 `GPU`/`GPU.n`，强制 FP32，不配置 CPU 候选设备。
- `auto` 优先 CUDA，不可用时再选 OpenVINO GPU；两者都不可用就终止启动。

OpenCV 解码和 NumPy 小规模余弦匹配在 CPU 上运行；它们不是占主要计算量的模型推理，也不是神经网络 CPU fallback。

## 离线运行边界

正式包使用 PyInstaller `onedir`，收集 Python、Qt、OpenCV、ONNX Runtime CUDA、OpenVINO 和 NVIDIA 运行 DLL，并把固定哈希模型放进只读资源目录。包内诊断会校验模型并在当前选中的 GPU 后端执行真实推理。

NVIDIA 显卡驱动必须由目标设备镜像预装。驱动不能由面向多种机器的普通应用包安全替代；这与不需要 CUDA Toolkit/cuDNN 预装并不冲突。

## 阈值与评估

默认阈值只是可操作的起点，不是生产承诺。正式部署应收集目标摄像头的：

- 已录入人员的正样本（日/夜、距离、侧脸、眼镜/口罩）；
- 未录入人员的大量负样本；
- 不同压缩码率、快门、焦距和安装高度。

然后按业务可接受的 FAR（误认率）选取相似度和 margin，再报告 TAR/TPIR。安防或身份决策不应仅依赖人脸识别；应保留人工复核或第二因素。

## 明确不做的事

- **不对过小/过糊人脸强猜**：原始像素中没有的身份信息无法被可靠恢复。
- **不把生成式人脸超分辨率放进识别链**：它可能生成错误的五官细节，造成身份偏移和误认。
- **不声称生物检测**：监控视频没有主动配合，此版不具备防照片/屏幕攻击的活体检测。

## 主要一手资料

- [LVFace 官方仓库](https://github.com/bytedance/LVFace)
- [LVFace ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/html/You_LVFace_Progressive_Cluster_Optimization_for_Large_Vision_Models_in_Face_ICCV_2025_paper.html)
- [AdaFace 官方仓库](https://github.com/mk-minchul/AdaFace)
- [CVLFace / KP-RPE 官方工具链](https://github.com/mk-minchul/CVLface)
- [InsightFace SCRFD 官方实现](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [AuraFace-v1 模型卡](https://huggingface.co/fal/AuraFace-v1)
