"""程序入口：GPU 强制检查 → 模型检查/下载 → 启动 GUI。

GPU 不可用时直接报错退出，绝不回退 CPU（CPU 太慢无法满足视频实时分析）。
"""
from __future__ import annotations

import sys
import traceback


def _fatal(msg: str, title: str = "无法启动") -> None:
    """尽量弹图形错误框，失败则打印到终端，最终以非零码退出。"""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, title, msg)
    except Exception:
        print(f"[FATAL] {title}: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    from face_match import config, gpu, models as models_mod

    # 1) 模型文件检查（GPU 检查前做，方便先补齐模型）
    if not models_mod.models_ready():
        rc = _prompt_download()
        if rc != 0:
            sys.exit(rc)

    # 2) GPU 强制检查：集成 pip CUDA 库 + 实际创建 CUDA 推理会话
    try:
        det_path = str(models_mod.model_path(config.DETECTOR_MODEL))
        gpu_info = gpu.ensure_gpu(det_path)
    except gpu.GPUUnavailableError as e:
        _fatal(str(e), "GPU 不可用")
        return
    except Exception as e:  # noqa: BLE001
        _fatal(f"GPU 初始化异常: {e}\n{traceback.format_exc(limit=3)}", "GPU 不可用")
        return

    # 3) 启动 GUI
    from PySide6.QtWidgets import QApplication
    from face_match.database import PersonDB
    from face_match.gui.main_window import MainWindow
    from face_match.gui.widgets import ModelHub

    app = QApplication(sys.argv)
    app.setApplicationName("FaceMatch")
    db = PersonDB()
    hub = ModelHub(gpu.cuda_providers())
    win = MainWindow(db, hub, gpu_info)
    win.show()
    sys.exit(app.exec())


def _prompt_download() -> int:
    """模型缺失时询问是否下载。返回 0 继续。"""
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog
    from face_match import models as models_mod

    app = QApplication.instance() or QApplication(sys.argv)
    ans = QMessageBox.question(
        None, "需要下载模型",
        "首次运行需要下载人脸检测/识别模型（约 290MB，来自 InsightFace 官方发布）。\n"
        "是否现在下载？\n\n（也可手动下载 buffalo_l.zip 解压到 data/models/ 目录）")
    if ans != QMessageBox.StandardButton.Yes:
        return 1

    dlg = QProgressDialog("正在下载模型…", "取消", 0, 100)
    dlg.setWindowTitle("下载模型")
    dlg.setMinimumDuration(0)
    dlg.setValue(0)

    class _W(QThread):
        progress = Signal(int)
        failed = Signal(str)

        def run(self) -> None:
            try:
                models_mod.download_models(
                    lambda done, total: self.progress.emit(
                        int(done * 100 / total) if total else 0))
            except Exception as e:  # noqa: BLE001
                self.failed.emit(str(e))

    w = _W()
    w.progress.connect(dlg.setValue)
    w.failed.connect(lambda m: (dlg.cancel(), _fatal(f"模型下载失败: {m}", "下载失败")))
    w.start()
    dlg.exec()
    if dlg.wasCanceled() and w.isRunning():
        w.terminate()
        return 1
    w.wait()
    if not models_mod.models_ready():
        _fatal("模型文件不完整，请检查网络后重试", "下载失败")
    return 0


if __name__ == "__main__":
    main()
