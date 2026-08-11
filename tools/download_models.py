"""预下载 insightface 模型包（antelopev2），供一键安装脚本调用。

模型会下载到 data/models/ 下，之后运行无需联网；
也可以把整个 data/models 目录拷贝到离线机器上直接使用。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402  # 设置 INSIGHTFACE_HOME 必须早于 insightface 导入


def main() -> None:
    from insightface.utils import storage

    config.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"下载模型包 {config.MODEL_NAME} 到 {config.MODEL_ROOT} ...")
    path = storage.ensure_available(
        "models", config.MODEL_NAME, root=str(config.MODEL_ROOT)
    )
    print(f"模型就绪: {path}")


if __name__ == "__main__":
    main()
