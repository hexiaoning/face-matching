# Face Matching

监控视频人脸比对系统。针对监控摄像头画面**模糊、非正脸**的场景设计，
将视频中出现的人脸与预录人员库比对，给出识别到的人员及置信度分数。

## 功能

- **人员库管理**：录入人员姓名、身份证号，每人 1~多张照片（清晰正脸照效果最好）。
- **视频分析**：打开本地视频文件，或接入视频流（RTSP / 摄像头编号）。
- **实时比对**：视频画面实时标注识别结果（绿色=已识别并显示姓名分数，黄色=陌生人，灰色=采样中）。
- **结果面板**：汇总识别到的人员、最高分数、首次/最近出现时间。
- **可调参数**：匹配阈值、跳帧数，全部图形界面鼠标操作。

## 针对监控画质的三层优化（SOTA 开源方案）

1. **模型层**：SCRFD 人脸检测 + ArcFace 识别（InsightFace buffalo_l，开源 SOTA），
   本身对小脸、模糊、大姿态鲁棒；识别模型可替换为 AdaFace（专为低质量人脸设计）。
2. **帧层**：对每帧人脸做质量评分（清晰度 Laplacian、姿态 yaw/pitch、人脸尺寸、
   检测置信度），过糊、过侧（yaw > 55°）、过小的帧直接过滤。
3. **轨迹层**：同一个人的多帧按 track 聚合，取质量 Top-K 帧的特征加权融合，
   单帧模糊/侧脸被同 track 的好帧纠正，显著降低误识别。

## 支持环境

| 项目 | 配置 |
| --- | --- |
| 设备 | 华硕 ROG G22CH 台式机 |
| CPU | Intel i7-14700KF |
| 显卡 | RTX 4070 SUPER 12GB |
| 内存 | 64GB |
| 系统 | Windows 11 64 位 |

**必须使用 NVIDIA GPU，GPU 不可用时程序报错退出**（CPU 太慢，不满足视频分析需求）。
目标设备无需安装 CUDA Toolkit —— CUDA 运行时库通过 pip 依赖
（`onnxruntime-gpu` + `nvidia-cuda-runtime-cu12` + `nvidia-cudnn-cu12`）自动集成，
仅需安装好 NVIDIA 显卡驱动。

## 快速开始（Windows）

1. 安装 Python 3.10+（勾选 Add to PATH）并安装最新 NVIDIA 驱动。
2. 双击 `install.bat` 一键安装依赖（含 CUDA 库，约 1.5GB）。
3. 双击 `run.bat` 启动；首次运行会引导下载人脸模型（约 290MB）。
4. 在「人员库管理」页录入人员与照片，在「视频分析」页打开视频开始比对。

也可手动执行：

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m face_match.app
```

## 识别分数说明

分数为余弦相似度（0~1），越高越可信。默认阈值 0.35：

- **≥ 0.45**：基本可确认是本人
- **0.35~0.45**：疑似本人，建议结合多张库照片提高分数
- **< 0.35**：判定为陌生人

库里同一人多录几张不同角度/清晰度的照片，可明显提高识别分数。

## 开发/测试

```bash
pip install onnxruntime opencv-python-headless numpy pillow pytest
python -m pytest tests/   # CPU 下验证 pipeline 逻辑（含合成监控视频端到端测试）
```

技术方案详见 [DESIGN.md](DESIGN.md)。
