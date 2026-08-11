# Third-party notices

离线包会收集各 Python wheel 自带的 `dist-info` 元数据和许可证文件。构建/交付人员仍需审核最终 `SHA256SUMS.txt` 和以下模型授权范围。

| 组件 | 用途 | 上游 / 许可提示 |
| --- | --- | --- |
| ONNX Runtime GPU | CUDA 推理 | Microsoft MIT |
| PySide6 / Qt | 桌面 GUI | LGPLv3/GPLv3/商业许可；交付前按所选 Qt 许可履行义务 |
| OpenCV | 视频与图像处理 | Apache-2.0 |
| NumPy | 数值计算 | BSD-3-Clause |
| NVIDIA CUDA/cuDNN/cuBLAS wheels | GPU 运行库 | NVIDIA SDK/EULA 条款 |
| SCRFD-10G | 人脸检测 | 参见 InsightFace/AuraFace 模型来源和适用条款 |
| LVFace-B Glint360K | 高精度人脸特征 | 代码 MIT；官方预训练权重仅限非商业研究，商用必须另获授权 |
| AuraFace-v1 | 可替代人脸特征 | Apache-2.0 模型卡；部署方仍需完成数据与用途合规审核 |

构建脚本不会绕过 LVFace 权重提示：包含 `lvface-b` 时必须显式传入 `-AcceptResearchWeights`。生物特征和身份证号属于敏感个人信息，模型许可证不等于取得数据处理或安防使用授权。
