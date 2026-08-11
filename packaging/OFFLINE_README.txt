Face Matching 0.2.1 - Windows 11 离线包
=========================================

使用方法：
1. 目标机器安装支持 RTX 4070 SUPER 的 NVIDIA 显卡驱动。
2. 将整个 FaceMatching 文件夹复制到目标机器；不要只复制 exe。
3. 双击“GPU诊断.bat”，确认 gpu_ready 和 inference_ready 都为 true。
4. 双击“启动.bat”。程序、Python、第三方库、CUDA/cuDNN DLL 和模型都已包含，
   运行期间不下载任何内容，也不需要安装 Python 或 CUDA Toolkit。

数据默认保存在：%LOCALAPPDATA%\FaceMatching
包括人员数据库、录入照片和运行数据。升级程序时不要删除该目录。

唯一未随包分发的运行依赖是 NVIDIA 显卡驱动。应用检测不到
CUDAExecutionProvider 或真实 GPU 推理失败时会拒绝启动，不会退回 CPU。

默认预训练模型权重只允许非商业研究使用。商用部署前请替换为获得相应授权的
ONNX 模型，并重新录入人脸与校准相似度阈值。
