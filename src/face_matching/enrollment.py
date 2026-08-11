from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np

from .database import Database
from .domain import EnrollmentReport, PreparedPhoto
from .face_engine import FaceEngine


class EnrollmentError(ValueError):
    pass


def read_image(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error) as exc:
        raise EnrollmentError(f"无法读取照片：{path.name}") from exc
    if image is None:
        raise EnrollmentError(f"无法读取照片：{path.name}")
    return image


class EnrollmentService:
    MIN_ENROLLMENT_QUALITY = 0.25
    RECOMMENDED_ENROLLMENT_QUALITY = 0.48

    def __init__(self, database: Database, engine: FaceEngine, gallery_dir: Path) -> None:
        self.database = database
        self.engine = engine
        self.gallery_dir = gallery_dir
        self.gallery_dir.mkdir(parents=True, exist_ok=True)

    def create_person(
        self, name: str, id_number: str, source_paths: list[Path]
    ) -> EnrollmentReport:
        self._validate_metadata(name, id_number, source_paths)
        prepared, created_files, warnings = self._prepare_photos(source_paths)
        try:
            person_id = self.database.add_person(name, id_number, prepared)
        except Exception:
            self._remove_files(created_files)
            raise
        return EnrollmentReport(person_id=person_id, warnings=warnings)

    def update_person(
        self,
        person_id: int,
        name: str,
        id_number: str,
        new_source_paths: list[Path],
        removed_photo_ids: list[int],
    ) -> EnrollmentReport:
        if not name.strip() or not id_number.strip():
            raise EnrollmentError("姓名和身份证号不能为空")
        prepared, created_files, warnings = self._prepare_photos(new_source_paths)
        try:
            removed_paths = self.database.update_person(
                person_id, name, id_number, prepared, removed_photo_ids
            )
        except Exception:
            self._remove_files(created_files)
            raise
        self._remove_gallery_files(removed_paths)
        return EnrollmentReport(person_id=person_id, warnings=warnings)

    def delete_person(self, person_id: int) -> None:
        removed_paths = self.database.delete_person(person_id)
        self._remove_gallery_files(removed_paths)

    def absolute_photo_path(self, relative_path: str) -> Path:
        candidate = (self.gallery_dir / relative_path).resolve()
        root = self.gallery_dir.resolve()
        if candidate.parent != root:
            raise EnrollmentError("人员库中存在不安全的照片路径")
        return candidate

    def _remove_gallery_files(self, relative_paths: list[str]) -> None:
        safe_paths: list[Path] = []
        for relative_path in relative_paths:
            try:
                safe_paths.append(self.absolute_photo_path(relative_path))
            except EnrollmentError:
                continue
        self._remove_files(safe_paths)

    def _prepare_photos(
        self, source_paths: list[Path]
    ) -> tuple[list[PreparedPhoto], list[Path], list[str]]:
        prepared: list[PreparedPhoto] = []
        created_files: list[Path] = []
        warnings: list[str] = []
        try:
            for source in source_paths:
                source = Path(source)
                image = read_image(source)
                observations = self.engine.analyze(image, threshold=0.45)
                if not observations:
                    raise EnrollmentError(f"{source.name}：未检测到人脸")
                if len(observations) != 1:
                    raise EnrollmentError(
                        f"{source.name}：检测到 {len(observations)} 张人脸，请使用仅含一个人的照片"
                    )
                face = observations[0]
                face_width = float(face.bbox[2] - face.bbox[0])
                face_height = float(face.bbox[3] - face.bbox[1])
                if min(face_width, face_height) < 40:
                    raise EnrollmentError(f"{source.name}：人脸小于 40 像素，不适合作为底库照片")
                if face.quality < self.MIN_ENROLLMENT_QUALITY:
                    raise EnrollmentError(
                        f"{source.name}：照片质量过低（{face.quality:.2f}），请换用更清晰照片"
                    )
                if face.quality < self.RECOMMENDED_ENROLLMENT_QUALITY:
                    warnings.append(
                        f"{source.name} 质量一般（{face.quality:.2f}），建议补充清晰正脸或侧脸照片"
                    )
                if face.embedding is None:
                    raise EnrollmentError(f"{source.name}：特征提取失败")
                extension = source.suffix.lower()
                if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    extension = ".jpg"
                target_name = f"{uuid.uuid4().hex}{extension}"
                target = self.gallery_dir / target_name
                if source.suffix.lower() == extension:
                    shutil.copy2(source, target)
                else:
                    ok, encoded = cv2.imencode(extension, image)
                    if not ok:
                        raise EnrollmentError(f"{source.name}：无法转换为受支持的图片格式")
                    with target.open("wb") as stream:
                        stream.write(encoded.tobytes())
                created_files.append(target)
                prepared.append(
                    PreparedPhoto(
                        path=target_name,
                        source_name=source.name,
                        quality=face.quality,
                        embedding=face.embedding,
                    )
                )
        except Exception:
            self._remove_files(created_files)
            raise
        return prepared, created_files, warnings

    @staticmethod
    def _validate_metadata(name: str, id_number: str, photos: list[Path]) -> None:
        if not name.strip():
            raise EnrollmentError("姓名不能为空")
        if not id_number.strip():
            raise EnrollmentError("身份证号不能为空")
        if not photos:
            raise EnrollmentError("请至少选择一张照片")

    @staticmethod
    def _remove_files(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
