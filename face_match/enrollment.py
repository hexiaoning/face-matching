from __future__ import annotations

import shutil
import sqlite3
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from face_match.config import MODEL_VERSION
from face_match.database import FaceDatabase, NewPhoto
from face_match.errors import EnrollmentError
from face_match.identity import validate_identity
from face_match.vision.alignment import align_face
from face_match.vision.detector import ScrfdDetector
from face_match.vision.embedder import LvFaceEmbedder
from face_match.vision.quality import assess_face_quality

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EnrollmentService:
    def __init__(
        self,
        database: FaceDatabase,
        detector: ScrfdDetector,
        embedder: LvFaceEmbedder,
        photo_root: Path,
    ) -> None:
        self.database = database
        self.detector = detector
        self.embedder = embedder
        self.photo_root = photo_root.resolve()
        self.photo_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        import cv2

        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError as exc:
            raise EnrollmentError(f"无法读取照片 {path.name}：{exc}") from exc
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise EnrollmentError(f"不是有效图片或格式不支持：{path.name}")
        return image

    def _extract(self, photo_paths: Sequence[Path]) -> list[tuple[Path, np.ndarray, float]]:
        if not photo_paths:
            raise EnrollmentError("至少选择一张人员照片")
        aligned_faces: list[np.ndarray] = []
        qualities: list[float] = []
        normalized_paths: list[Path] = []
        for raw_path in photo_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise EnrollmentError(f"照片不存在：{path}")
            if path.suffix.lower() not in _ALLOWED_SUFFIXES:
                raise EnrollmentError(f"不支持的照片格式：{path.name}")
            image = self._read_image(path)
            detections = self.detector.detect(image, score_threshold=0.55)
            if not detections:
                raise EnrollmentError(f"照片 {path.name} 中没有检测到人脸")
            if len(detections) != 1:
                raise EnrollmentError(
                    f"照片 {path.name} 中检测到 {len(detections)} 张人脸；录入照片必须只含一人"
                )
            detection = detections[0]
            try:
                aligned = align_face(image, detection.landmarks)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise EnrollmentError(f"照片 {path.name} 的人脸无法可靠对齐") from exc
            quality = assess_face_quality(aligned, detection, image.shape)
            if quality.overall < 0.22:
                raise EnrollmentError(
                    f"照片 {path.name} 质量过低（{quality.overall:.2f}）。"
                    "请使用更清晰、脸部更大的照片；建议同时录入正面和左右侧面照。"
                )
            normalized_paths.append(path)
            aligned_faces.append(aligned)
            qualities.append(quality.overall)
        embeddings = self.embedder.embed(aligned_faces)
        return [
            (path, embedding, quality)
            for path, embedding, quality in zip(normalized_paths, embeddings, qualities)
        ]

    def _copy_new_photos(
        self, extracted: Sequence[tuple[Path, np.ndarray, float]], folder_name: str
    ) -> tuple[list[NewPhoto], list[Path]]:
        folder = self.photo_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        records: list[NewPhoto] = []
        created: list[Path] = []
        try:
            for source, embedding, quality in extracted:
                suffix = source.suffix.lower()
                target = folder / f"{uuid.uuid4().hex}{suffix}"
                shutil.copy2(source, target)
                created.append(target)
                records.append(NewPhoto(target, source.name, embedding, quality, MODEL_VERSION))
        except OSError as exc:
            self._remove_files(created)
            raise EnrollmentError(f"复制人员照片失败：{exc}") from exc
        return records, created

    def add_person(self, name: str, id_number: str, photo_paths: Sequence[Path]) -> int:
        name, id_number = validate_identity(name, id_number)
        extracted = self._extract(photo_paths)
        folder_name = f"person-{uuid.uuid4().hex}"
        records, created = self._copy_new_photos(extracted, folder_name)
        try:
            return self.database.add_person(name, id_number, records)
        except sqlite3.IntegrityError as exc:
            self._remove_files(created)
            raise EnrollmentError("身份证号已存在，不能重复录入") from exc
        except Exception:
            self._remove_files(created)
            raise

    def update_person(
        self,
        person_id: int,
        name: str,
        id_number: str,
        new_photo_paths: Sequence[Path],
        remove_photo_ids: Sequence[int],
    ) -> None:
        name, id_number = validate_identity(name, id_number)
        extracted = self._extract(new_photo_paths) if new_photo_paths else []
        records, created = self._copy_new_photos(extracted, f"person-{person_id}")
        try:
            removed = self.database.update_person(
                person_id, name, id_number, records, remove_photo_ids
            )
        except sqlite3.IntegrityError as exc:
            self._remove_files(created)
            raise EnrollmentError("身份证号已被其他人员使用") from exc
        except Exception:
            self._remove_files(created)
            raise
        self._remove_files(removed)

    def delete_person(self, person_id: int) -> None:
        removed = self.database.delete_person(person_id)
        self._remove_files(removed)

    def _remove_files(self, paths: Sequence[Path]) -> None:
        parents: set[Path] = set()
        for path in paths:
            try:
                resolved = path.resolve()
                if not resolved.is_relative_to(self.photo_root):
                    continue
                resolved.unlink(missing_ok=True)
                parents.add(resolved.parent)
            except OSError:
                continue
        for parent in parents:
            try:
                if parent != self.photo_root and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                pass
