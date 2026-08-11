from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import AppConfig
from .gpu import assert_gpu_available
from .models import PROFILES, available_profiles, download_profile, required_paths
from .paths import config_path, is_frozen_app


def _write_report(path: Path | None, payload: dict[str, object]) -> None:
    value = json.dumps(payload, ensure_ascii=False, indent=2)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    if sys.stdout is not None:
        print(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU-only 视频人脸比对")
    parser.add_argument("--check-gpu", action="store_true", help="检查 CUDA/OpenVINO GPU 并退出")
    parser.add_argument("--download-models", action="store_true", help="下载模型并退出")
    parser.add_argument("--verify-models", action="store_true", help="校验模型 SHA-256 并退出")
    parser.add_argument("--diagnose", action="store_true", help="校验模型并执行真实 GPU 推理")
    parser.add_argument("--report", type=Path, help="把诊断 JSON 写入指定文件")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="覆盖配置中的模型档")
    args = parser.parse_args(argv)
    config = AppConfig.load()
    if args.profile:
        config.model_profile = args.profile
    elif is_frozen_app() and not config_path().exists():
        bundled = available_profiles()
        if bundled:
            config.model_profile = "lvface-b" if "lvface-b" in bundled else bundled[0]
    if args.download_models:
        if is_frozen_app():
            print("离线便携版不允许联网下载模型；请重新生成包含所需模型的安装包。", file=sys.stderr)
            return 2
        download_profile(config.model_profile)
        print(f"模型 {config.model_profile} 已就绪")
        return 0
    if args.verify_models:
        try:
            paths = required_paths(config.model_profile, verify_hash=True)
            _write_report(args.report, {
                "ok": True,
                "profile": config.model_profile,
                "models": [str(path) for path in paths],
            })
            return 0
        except Exception as exc:
            _write_report(args.report, {"ok": False, "error": str(exc)})
            return 2
    if args.diagnose:
        try:
            from .diagnostics import run_diagnostics

            _write_report(
                args.report,
                run_diagnostics(
                    config.model_profile,
                    backend=config.gpu_backend,
                    device_id=config.gpu_device_id,
                    mirror_augmentation=config.mirror_augmentation,
                ),
            )
            return 0
        except Exception as exc:
            _write_report(args.report, {
                "ok": False,
                "profile": config.model_profile,
                "error": str(exc),
            })
            return 2
    if args.check_gpu:
        try:
            print(assert_gpu_available(config.gpu_backend, config.gpu_device_id))
            return 0
        except Exception as exc:
            print(f"GPU 不可用: {exc}", file=sys.stderr)
            return 2

    from PySide6.QtWidgets import QApplication
    from .gui.main_window import MainWindow

    application = QApplication(sys.argv[:1])
    application.setApplicationName("Face Matching")
    application.setStyle("Fusion")
    window = MainWindow(config)
    if not window.initialize():
        return 2
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
