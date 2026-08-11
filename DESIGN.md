# 技术方案设计

## 目标

监控视频（可能模糊、非正脸）中的人脸与预录人员库（姓名 + 身份证号 + 1~多张照片）比对，给出识别人员和分数。图形界面、全鼠标操作。**必须使用 GPU，不可用则报错退出**。

## 方案选型（开源 SOTA）

| 模块 | 选型 | 理由 |
| --- | --- | --- |
| 人脸检测 | **SCRFD**（InsightFace，det_10g） | 当前开源检测 SOTA 之一，对小脸/模糊/遮挡鲁棒，速度快 |
| 人脸识别 | **ArcFace (buffalo_l, w600k_r50)** 默认；**AdaFace** 可切换 | ArcFace 是工业界事实标准；AdaFace (CVPR'22) 专为低质量/模糊人脸设计，特征范数即质量 |
| 推理引擎 | **ONNX Runtime GPU**（CUDAExecutionProvider） | 跨平台，Windows 上可通过 pip 的 `nvidia-cuda-runtime-cu12` / `nvidia-cudnn-cu12` 集成 CUDA 运行时，无需手动安装 CUDA Toolkit |
| 质量评估 | 人脸尺寸 + Laplacian 清晰度 + 关键点估计姿态(yaw/pitch) + 检测置信度 | 监控视频必须过滤/加权低质量帧 |
| 多帧聚合 | IoU tracker 分 track，每 track 按质量取 Top-K 帧 embedding 加权平均 | 多帧融合是监控场景提升准确率的关键工程手段，显著缓解单帧模糊/侧脸 |
| 匹配 | 余弦相似度；每人多张照片取最大相似度；阈值可调 | 简单、可解释、分数即置信度 |
| 人员库 | SQLite（姓名、身份证号、照片路径、embedding 缓存） | 零部署、随程序携带 |
| GUI | PySide6 (Qt) | 全鼠标操作、跨平台、视频播放组件成熟 |

针对"模糊/非正面"的三层对策：
1. **模型层**：SCRFD + ArcFace/AdaFace 本身对低质量、大姿态鲁棒。
2. **帧层**：质量评分过滤掉过糊、过侧（yaw > 阈值）的帧。
3. **轨迹层**：同一人的多帧 embedding 质量加权融合，单帧差帧被好帧"拉正"。

## 目录结构

```
face_match/
├── app.py                 # 入口：GPU 检查 → 启动 GUI
├── gpu.py                 # CUDA DLL 集成 + GPU 强制检查
├── config.py              # 路径、常量、阈值默认值
├── models.py              # 模型下载/加载管理
├── detector.py            # SCRFD ONNX 检测器
├── recognizer.py          # ArcFace/AdaFace ONNX 识别器（112x112 对齐）
├── quality.py             # 清晰度/姿态/综合质量评分
├── tracker.py             # IoU tracker + track 级 embedding 聚合
├── pipeline.py            # 视频帧 → detect → track → 识别 → 匹配
├── database.py            # SQLite 人员库 + embedding 索引
├── matcher.py             # 相似度匹配
└── gui/
    ├── main_window.py     # 主窗口（人员库 / 视频分析 两个页签）
    ├── persons_page.py    # 人员管理：增删改、多照片
    ├── video_page.py      # 视频播放、实时检测框、结果面板
    └── widgets.py         # 视频画面控件、结果列表等
install.bat                # Windows 一键安装
run.bat                    # Windows 一键启动
requirements.txt
tests/                     # 单元测试（CPU provider 下验证逻辑）
```

## GPU 强制策略

启动时：
1. `gpu.py` 把 pip 安装的 `nvidia-*` 包中的 DLL 目录加入 DLL 搜索路径（Windows）。
2. 检查 `onnxruntime.get_available_providers()` 是否含 `CUDAExecutionProvider`，且能成功创建 session。
3. 不满足 → 弹错误对话框说明原因 → 退出（非零退出码）。绝不回退 CPU。

（单元测试通过显式传 `providers=["CPUExecutionProvider"]` 绕过，仅用于开发机验证逻辑。）
