# Face Matching

视频人脸比对系统（GPU 版）。面向监控摄像头场景，针对模糊、非正面人脸做了专门优化。

## 功能

- 人员库管理：每人 = 姓名 + 身份证号 + 1~多张照片（SQLite 存储，照片提取 512 维特征）。
- 视频输入：本地视频文件 / USB 摄像头 / RTSP 网络摄像头。
- 视频中的人脸与人员库实时比对，画面标注姓名与相似度分数，命中记录自动留存抓拍图。
- 全图形界面、鼠标操作。

## 技术方案（监控场景 SOTA 开源路线）

| 环节 | 方案 |
| --- | --- |
| 人脸检测 | SCRFD（InsightFace），小脸/遮挡表现优秀 |
| 特征识别 | antelopev2 模型包（GlintR100 + ArcFace），基于 Glint360K，非约束场景开源最强 |
| 模糊/侧脸应对 | ① 质量门控：清晰度（拉普拉斯方差）+ 关键点偏航角 + 人脸尺寸，过滤劣质帧；② IoU 多目标跟踪；③ 多帧质量加权特征融合（同一目标取质量最高的 8 帧特征加权平均后再比对） |
| 推理引擎 | ONNX Runtime GPU：优先 CUDA，备选 DirectML；**无 GPU 直接报错退出，绝不回退 CPU** |
| 特征比对 | 归一化余弦相似度，一人多照片取最大值；阈值界面可调（默认 0.45） |

## 支持环境

| 项目 | 配置 |
| --- | --- |
| 设备 | 华硕 ROG G22CH 台式机 |
| CPU | Intel i7-14700KF |
| 显卡 | RTX 4070 SUPER 12GB |
| 内存 | 64GB |
| 系统 | Windows 11 64 位 |

目标设备无需安装 CUDA 工具包：CUDA/cuDNN 运行库以 pip 包（`nvidia-*-cu12`）随依赖自动安装，模型文件下载到项目内 `data/models/`。

## 一键运行

1. 安装 Python 3.10+（勾选 Add to PATH）和 NVIDIA 显卡驱动。
2. 双击 `run.bat`：自动创建虚拟环境 → 安装依赖 → 下载模型 → 启动界面。

首次运行需要联网下载模型，之后可离线使用；拷贝 `data/models/` 即可部署到离线机器。

## 使用说明

1. 左侧【添加人员】：填写姓名、18 位身份证号，选择 1~N 张照片（建议清晰正面登记照，可多角度多张）。
2. 右侧选择视频源（文件 / 摄像头编号如 `0` / RTSP 地址），点击【开始】。
3. 拖动【相似度阈值】滑块调整松紧：调低识别更多但可能误报，调高更严格。识别中可实时调整。
4. 命中记录表格保存时间、姓名、身份证号、分数和抓拍图，双击可打开抓拍图目录。

## 命令行选项

```bash
python -m app.main --backend auto      # auto|cuda|directml，默认 auto
python -m app.main --model buffalo_l   # 换用更轻量的模型包
```

## 目录结构

```
app/
  config.py        # 全部可调参数（质量门控、跟踪、阈值等）
  gpu.py           # CUDA 运行库加载与 GPU 强制检查
  engine.py        # SCRFD 检测 + GlintR100 特征提取（GPU）
  quality.py       # 清晰度 / 姿态 / 尺寸质量评估
  tracker.py       # IoU 跟踪 + 多帧质量加权特征融合
  gallery.py       # 特征库比对（一人多照片取最大相似度）
  db.py            # SQLite 人员库 / 照片 / 特征 / 命中记录
  worker.py        # 视频处理线程（检测->跟踪->融合->比对->标注）
  gui/             # PyQt6 界面
  main.py          # 入口（GPU 初始化失败即报错退出）
tools/download_models.py  # 模型预下载
run.bat                   # Windows 一键安装运行
```
