Face Matching 0.2.2 - Windows 11 离线包
=========================================

使用方法：
1. CUDA 包：安装支持 RTX 5070 的 NVIDIA 驱动。
   DirectML 包：安装最新 Intel 核显驱动并确认 DirectX 12 可用。
2. 将整个 FaceMatching 文件夹复制到目标机器；不要只复制 exe。
3. 双击“GPU诊断.bat”，确认 selected_provider 与包类型一致，且
   gpu_ready 和 inference_ready 都为 true。
4. 双击“启动.bat”。程序、Python、第三方库、GPU 运行 DLL 和模型都已包含，
   运行期间不下载任何内容，也不需要安装 Python 或 CUDA Toolkit。

数据默认保存在：%LOCALAPPDATA%\FaceMatching
包括人员数据库、录入照片和运行数据。升级程序时不要删除该目录。

未随包分发的运行依赖只有相应显卡驱动。应用检测不到 CUDAExecutionProvider
或 DmlExecutionProvider、或真实 GPU 推理失败时会拒绝启动，不会退回 CPU。

这是内部系统，不提供身份证号或视频画面脱敏，界面显示完整原始信息。

默认预训练模型权重只允许非商业研究使用。商用部署前请替换为获得相应授权的
ONNX 模型，并重新录入人脸与校准相似度阈值。
