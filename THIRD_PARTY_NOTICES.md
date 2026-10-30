# Third-party notices

离线包会收集各 Python wheel 自带的 `dist-info` 元数据和许可证文件。构建/交付人员仍需审核最终 `SHA256SUMS.txt` 和以下模型授权范围。

| 组件 | 用途 | 上游 / 许可提示 |
| --- | --- | --- |
| ONNX Runtime GPU | NVIDIA CUDA 推理 | Microsoft MIT |
| OpenVINO | Intel GPU 推理 | Apache-2.0 |
| PySide6 / Qt | 桌面 GUI | LGPLv3/GPLv3/商业许可；交付前按所选 Qt 许可履行义务 |
| OpenCV | 视频与图像处理 | Apache-2.0 |
| RT-DETRv2-R50 / ONNX Community 导出 | 人和车辆的类别、位置及置信度检测 | 作者 lyuwenyu / PekingU，Apache-2.0；ONNX 导出固定版本 1699ea019e5570bb7283d80ee9a9e42d6245436f；许可证随包保存在 licenses/RT-DETR-LICENSE.txt |
| NumPy | 数值计算 | BSD-3-Clause |
| Chinese-CLIP ViT-B/16 | 中文衣着与车辆颜色语义检索 | OFA-Sys，MIT；FP32 ONNX转换来自 felixdu/chinese-clip-vit-base-patch16-onnx，固定版本47080d16c631d8416d2e6b155c59f8fd2c322e98；匹配词表来自Xenova固定版本f26904860903e70e050b8f48255e5f48401816e9。两编码器及词表均固定SHA-256，许可证随包保存在licenses/Chinese-CLIP-LICENSE.txt。 |
| NVIDIA CUDA/cuDNN/cuBLAS wheels | GPU 运行库 | NVIDIA SDK/EULA 条款 |
| SCRFD-10G | 人脸检测 | 参见 InsightFace/AuraFace 模型来源和适用条款 |
| LVFace-B Glint360K | 高精度人脸特征 | 代码 MIT；官方预训练权重仅限非商业研究，商用必须另获授权 |
| AuraFace-v1 | 可替代人脸特征 | Apache-2.0 模型卡；部署方仍需完成数据与用途合规审核 |

构建脚本不会绕过 LVFace 权重提示：包含 `lvface-b` 时必须显式传入 `-AcceptResearchWeights`。生物特征和身份证号属于敏感个人信息，模型许可证不等于取得数据处理或安防使用授权。


## Offline administrative region reference data

The 2023 province/city/county reference JSON and its WTFPL license are bundled under
`face_matching/resources`. Source: https://github.com/modood/Administrative-divisions-of-China
at commit c49d495b40ac73eb1a66f6eeae5f8fd10696f035. Dataset date: 2023-06-30.
The pinned source URL and SHA-256 are recorded in regions-source.json. Names are
historical references; unmatched historic address codes can be entered directly.
