# v2.1 架构与交付

## 识别链路

```text
视频 / 摄像头 / RTSP
  -> SCRFD-10G GPU 检测（默认 960，低阈值提升困难脸召回）
  -> 五点对齐 + 尺寸/清晰度/姿态/置信度质量评估
  -> LVFace-B GPU 特征（原图 + 水平镜像 TTA，按模型能力批处理）
  -> 运动预测 + IoU 跟踪
  -> 高质量 top-k + medoid 离群抑制的轨迹模板
  -> 人员级最佳样本 / top-k / 质量中心融合
  -> 相似度阈值 + 第一二名 margin + 连续新特征确认
  -> GUI 叠框、脱敏身份证号和命中事件
```

神经网络会话只注册 TensorRT（显式启用时）和 CUDA provider，同时设置
`session.disable_cpu_ep_fallback=1`。CUDA provider 缺失、模型加载失败或真实预热推理失败都会阻止启动。

## 离线边界

`build_offline_bundle.bat` 只在联网构建机运行。产物使用 PyInstaller one-folder，包含：

- Python 解释器和应用代码；
- PySide6、OpenCV、NumPy、ONNX Runtime GPU；
- 从 `nvidia` site-packages 收集的 CUDA 12、cuBLAS、cuDNN DLL；
- SCRFD 和 LVFace ONNX 模型；
- 依赖版本清单、全文件 SHA-256 清单、启动与 GPU 诊断脚本。

构建默认从冻结后的 EXE 运行两个模型的真实 GPU 自检。目标机不安装包、不访问网络，只依赖 NVIDIA 驱动。人员数据库和照片不写入程序目录，而保存在 `%LOCALAPPDATA%\FaceMatching`，升级离线包不会覆盖人员数据。

## 各候选实现的取舍

- 采用 `codex/sota-gpu-face-matching-d521` 的模块化 GUI、GPU 硬校验、LVFace/SCRFD 模型管理和诊断骨架。
- 采用 `codex/gpu-face-matching-gui` 的镜像 TTA、批大小兼容和样本/人员中心融合思路。
- 采用 `run20260806_0_1` 的 medoid 异常特征过滤，保留基线的运动预测与双重时序确认。
- 采用 `run20260806_0_2` 的 SCRFD 动态 anchor、双分类输出兼容和动态/固定输入检查。
- 吸收 `run20260806_0_3`、`run20260806_0_4` 的本地模型、冻结路径和困难视频端到端验证方向，并把“首次联网安装”升级为真正不依赖目标机 Python/网络的 one-folder 离线包。

默认值以 RTX 4070 SUPER 12GB 的效果优先部署为目标：检测输入为 960，检测每帧执行，识别每三帧采样且使用镜像 TTA。现场可用 `FACE_MATCHING_DETECTOR_SIZE=640` 或 `FACE_MATCHING_MIRROR_TTA=0` 换取速度，但正式阈值仍必须用目标摄像头数据校准。
