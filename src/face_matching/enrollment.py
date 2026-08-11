from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .database import EnrollmentSample
from .detector import FaceDetection
from .engine import FaceEngine
from .errors import EnrollmentError
from .image_io import read_image


ProgressCallback = Callable[[int, int, str], None]


def _select_enrollment_face(
    faces: Sequence[FaceDetection],
    min_face_size: int,
    filename: str,
) -> FaceDetection:
    if not faces:
        raise EnrollmentError(f"未检测到人脸：{filename}")
    ranked = sorted(
        faces,
        key=lambda face: (
            max(0.0, float(face.bbox[2] - face.bbox[0]))
            * max(0.0, float(face.bbox[3] - face.bbox[1]))
            * max(float(face.score), 0.0)
        ),
        reverse=True,
    )
    primary = ranked[0]
    primary_area = max(
        float(primary.bbox[2] - primary.bbox[0])
        * float(primary.bbox[3] - primary.bbox[1]),
        1.0,
    )
    primary_width = float(primary.bbox[2] - primary.bbox[0])
    primary_height = float(primary.bbox[3] - primary.bbox[1])
    if min(primary_width, primary_height) < min_face_size:
        raise EnrollmentError(
            f"人脸过小：{filename}（{min(primary_width, primary_height):.0f}px）"
        )
    # Ignore tiny background detections, but reject a second meaningful face
    # so the database cannot silently enroll the wrong identity.
    for face in ranked[1:]:
        width = float(face.bbox[2] - face.bbox[0])
        height = float(face.bbox[3] - face.bbox[1])
        if min(width, height) >= min_face_size and width * height >= primary_area * 0.25:
            raise EnrollmentError(f"照片包含多张明显人脸：{filename}")
    return primary


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
        face = _select_enrollment_face(faces, engine.config.min_face_size, path.name)
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
