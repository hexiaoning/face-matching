from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from .database import FaceDatabase, FaceSampleInput
from .engine import FaceEngine
from .errors import EnrollmentError
from .paths import data_dir
from .vision.io import read_image, write_jpeg


class EnrollmentService:
    def __init__(self, database: FaceDatabase, engine: FaceEngine, image_root: Path | None = None) -> None:
        self.database = database
        self.engine = engine
        self.image_root = image_root or (data_dir() / "face_images")
        self.image_root.mkdir(parents=True, exist_ok=True)

    def _prepare(self, person_id: str, photo_paths: list[str | Path]) -> tuple[list[FaceSampleInput], list[Path]]:
        if not photo_paths:
            raise EnrollmentError("请至少选择一张照片")
        destination_dir = self.image_root / person_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        samples: list[FaceSampleInput] = []
        created: list[Path] = []
        try:
            for source in photo_paths:
                source_path = Path(source)
                image = read_image(source_path)
                if image is None:
                    raise EnrollmentError(f"无法读取照片: {source_path.name}")
                try:
                    feature = self.engine.enrollment_feature(image)
                except EnrollmentError as exc:
                    raise EnrollmentError(f"{source_path.name}: {exc}") from exc
                destination = destination_dir / f"{uuid.uuid4().hex}.jpg"
                write_jpeg(destination, image)
                created.append(destination)
                samples.append(
                    FaceSampleInput(
                        image_path=str(destination),
                        embedding=feature.embedding,
                        model_id=self.engine.profile.model_id,
                        quality=feature.quality.total,
                    )
                )
            return samples, created
        except Exception:
            self._cleanup_files(created)
            try:
                destination_dir.rmdir()
            except OSError:
                pass
            raise

    @staticmethod
    def _cleanup_files(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def add_person(self, name: str, id_card: str, photo_paths: list[str | Path]) -> str:
        person_id = uuid.uuid4().hex
        samples, created = self._prepare(person_id, photo_paths)
        try:
            self.database.add_person(name, id_card, samples, person_id=person_id)
        except sqlite3.IntegrityError as exc:
            self._cleanup_files(created)
            try:
                (self.image_root / person_id).rmdir()
            except OSError:
                pass
            if "id_card" in str(exc).lower() or "unique" in str(exc).lower():
                raise EnrollmentError("该身份证号已经存在") from exc
            raise EnrollmentError(f"保存人员失败: {exc}") from exc
        self.engine.refresh_gallery()
        return person_id

    def add_photos(self, person_id: str, photo_paths: list[str | Path]) -> None:
        samples, created = self._prepare(person_id, photo_paths)
        try:
            self.database.add_samples(person_id, samples)
        except Exception:
            self._cleanup_files(created)
            raise
        self.engine.refresh_gallery()

    def delete_person(self, person_id: str) -> None:
        paths = self.database.delete_person(person_id)
        root = self.image_root.resolve()
        parents: set[Path] = set()
        for value in paths:
            path = Path(value).resolve()
            if path.is_relative_to(root):
                try:
                    path.unlink(missing_ok=True)
                    parents.add(path.parent)
                except OSError:
                    pass
        for parent in parents:
            try:
                parent.rmdir()
            except OSError:
                pass
        self.engine.refresh_gallery()
