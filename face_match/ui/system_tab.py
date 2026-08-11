from __future__ import annotations

from typing import Any

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from face_match.config import MODEL_VERSION
from face_match.model_manager import MODEL_LICENSE_NOTICE
from face_match.services import ApplicationServices


class SystemTab(QWidget):
    def __init__(self, services: ApplicationServices, parent: Any | None = None) -> None:
        super().__init__(parent)
        gpu_group = QGroupBox("GPU 状态")
        gpu_form = QFormLayout(gpu_group)
        gpu_form.addRow("设备：", QLabel(services.gpu.device_name))
        gpu_form.addRow("显存：", QLabel(services.gpu.memory_total))
        gpu_form.addRow("驱动：", QLabel(services.gpu.driver_version))
        gpu_form.addRow("执行器：", QLabel(services.gpu.provider))
        gpu_form.addRow("ONNX Runtime：", QLabel(services.gpu.runtime_version))
        strict = QLabel("CPU 推理回退：已禁用")
        strict.setObjectName("successLabel")
        gpu_form.addRow("安全检查：", strict)

        model_group = QGroupBox("模型与数据")
        model_form = QFormLayout(model_group)
        model_form.addRow("识别模型：", QLabel(f"LVFace-B Glint360K ({MODEL_VERSION})"))
        model_form.addRow("检测模型：", QLabel("InsightFace 10GF + 5 点关键点"))
        model_form.addRow("数据目录：", QLabel(str(services.paths.root)))
        model_form.addRow("数据库：", QLabel(str(services.paths.database)))
        open_button = QPushButton("打开本地数据目录")
        open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(services.paths.root)))
        )
        model_form.addRow("", open_button)

        license_label = QLabel(MODEL_LICENSE_NOTICE)
        license_label.setWordWrap(True)
        license_label.setObjectName("warningLabel")
        privacy = QLabel(
            "隐私提示：人员照片、身份证号与特征向量属于敏感生物识别信息。应用不会上传这些数据，"
            "但部署方仍应设置磁盘访问权限、留存期限和合法使用流程。"
        )
        privacy.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(gpu_group)
        layout.addWidget(model_group)
        layout.addWidget(license_label)
        layout.addWidget(privacy)
        layout.addStretch()
