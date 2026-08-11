from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .database import EnrollmentSample
from .engine import FaceEngine
from .errors import EnrollmentError
from .image_io import read_image


ProgressCallback = Callable[[int, int, str], None]


def prepare_enrollment_samples(
    paths: Sequence[str | Path],
    engine: FaceEngine,
    progress: ProgressCallback | None = None,
) -> list[EnrollmentSample]:
    if not paths:
        return []
    detections = []
    normalized_paths = [Path(path) for path in paths]
    for index, path in enumerate(normalized_paths, start=1):
        if progress:
            progress(index - 1, len(paths), path.name)
        image = read_image(path)
        faces = engine.detect(image)
        if not faces:
            raise EnrollmentError(f"未检测到人脸：{path.name}")
        if len(faces) != 1:
            raise EnrollmentError(f"照片必须且只能包含一张人脸：{path.name}（检测到 {len(faces)} 张）")
        face = faces[0]
        quality = float(face.quality.overall) if face.quality else 0.0
        if quality < engine.config.enrollment_min_quality:
            raise EnrollmentError(
                f"照片质量过低：{path.name}（{quality:.2f} < "
                f"{engine.config.enrollment_min_quality:.2f}）。请换用更清晰的照片。"
            )
        detections.append(face)
    embeddings = engine.embed([face.aligned for face in detections if face.aligned is not None])
    if len(embeddings) != len(detections):
        raise EnrollmentError("部分照片对齐失败")
    samples = [
        EnrollmentSample(path, embedding, float(face.quality.overall))
        for path, face, embedding in zip(normalized_paths, detections, embeddings, strict=True)
    ]
    if progress:
        progress(len(paths), len(paths), "完成")
    return samples
