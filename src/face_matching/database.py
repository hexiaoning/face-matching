from __future__ import annotations

import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .paths import database_path, photo_dir


@dataclass(frozen=True, slots=True)
class Person:
    id: str
    name: str
    id_card: str
    photo_count: int
    created_at: str


@dataclass(frozen=True, slots=True)
class EnrollmentSample:
    source_path: Path
    embedding: np.ndarray
    quality: float


@dataclass(frozen=True, slots=True)
class FaceSample:
    id: str
    person_id: str
    image_path: Path
    quality: float
    embedding_model: str
    created_at: str


@dataclass(frozen=True, slots=True)
class GallerySample:
    person_id: str
    name: str
    id_card: str
    embedding: np.ndarray
    quality: float


SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL CHECK(length(trim(name)) > 0),
    id_card TEXT NOT NULL UNIQUE CHECK(length(trim(id_card)) > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_samples (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    quality REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_face_samples_person ON face_samples(person_id);
CREATE INDEX IF NOT EXISTS idx_face_samples_model ON face_samples(embedding_model);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FaceDatabase:
    def __init__(self, path: Path | None = None, photos: Path | None = None) -> None:
        self.path = Path(path or database_path())
        self.photos = Path(photos or photo_dir())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.photos.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.executescript(SCHEMA)
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def list_people(self) -> list[Person]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT p.id, p.name, p.id_card, p.created_at, COUNT(s.id) AS photo_count
                FROM people p LEFT JOIN face_samples s ON s.person_id = p.id
                GROUP BY p.id ORDER BY p.name COLLATE NOCASE, p.created_at
                """
            ).fetchall()
        return [Person(**dict(row)) for row in rows]

    def get_person(self, person_id: str) -> Person | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT p.id, p.name, p.id_card, p.created_at, COUNT(s.id) AS photo_count
                FROM people p LEFT JOIN face_samples s ON s.person_id = p.id
                WHERE p.id = ? GROUP BY p.id
                """,
                (person_id,),
            ).fetchone()
        return Person(**dict(row)) if row else None

    def list_face_samples(self, person_id: str) -> list[FaceSample]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, person_id, image_path, quality, embedding_model, created_at
                FROM face_samples WHERE person_id = ?
                ORDER BY quality DESC, created_at
                """,
                (person_id,),
            ).fetchall()
        return [
            FaceSample(
                id=row["id"],
                person_id=row["person_id"],
                image_path=Path(row["image_path"]),
                quality=float(row["quality"]),
                embedding_model=row["embedding_model"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def add_person(
        self,
        name: str,
        id_card: str,
        samples: Sequence[EnrollmentSample],
        model_id: str,
    ) -> str:
        name, id_card = _validate_person(name, id_card)
        if not samples:
            raise ValueError("at least one face photo is required")
        person_id = str(uuid.uuid4())
        copied: list[Path] = []
        now = _utc_now()
        try:
            with self._transaction() as connection:
                connection.execute(
                    "INSERT INTO people(id, name, id_card, created_at, updated_at) VALUES(?,?,?,?,?)",
                    (person_id, name, id_card, now, now),
                )
                for sample in samples:
                    destination = self._copy_photo(person_id, sample.source_path)
                    copied.append(destination)
                    self._insert_sample(connection, person_id, destination, sample, model_id, now)
        except Exception:
            for path in copied:
                path.unlink(missing_ok=True)
            raise
        return person_id

    def update_person(
        self,
        person_id: str,
        name: str,
        id_card: str,
        new_samples: Sequence[EnrollmentSample],
        model_id: str,
    ) -> None:
        name, id_card = _validate_person(name, id_card)
        existing = self.get_person(person_id)
        if existing is None:
            raise KeyError(person_id)
        if existing.photo_count + len(new_samples) < 1:
            raise ValueError("at least one face photo is required")
        copied: list[Path] = []
        now = _utc_now()
        try:
            with self._transaction() as connection:
                connection.execute(
                    "UPDATE people SET name = ?, id_card = ?, updated_at = ? WHERE id = ?",
                    (name, id_card, now, person_id),
                )
                for sample in new_samples:
                    destination = self._copy_photo(person_id, sample.source_path)
                    copied.append(destination)
                    self._insert_sample(connection, person_id, destination, sample, model_id, now)
        except Exception:
            for path in copied:
                path.unlink(missing_ok=True)
            raise

    def _copy_photo(self, person_id: str, source: Path) -> Path:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        extension = source.suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            extension = ".jpg"
        person_dir = self.photos / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        destination = person_dir / f"{uuid.uuid4().hex}{extension}"
        shutil.copy2(source, destination)
        return destination

    @staticmethod
    def _insert_sample(
        connection: sqlite3.Connection,
        person_id: str,
        destination: Path,
        sample: EnrollmentSample,
        model_id: str,
        created_at: str,
    ) -> None:
        embedding = np.asarray(sample.embedding, dtype=np.float32).reshape(-1)
        connection.execute(
            """
            INSERT INTO face_samples(
                id, person_id, image_path, embedding, embedding_dim,
                embedding_model, quality, created_at
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()), person_id, str(destination), embedding.tobytes(),
                int(embedding.size), model_id, float(sample.quality), created_at,
            ),
        )

    def delete_person(self, person_id: str) -> None:
        with self._lock:
            rows = self._connection.execute(
                "SELECT image_path FROM face_samples WHERE person_id = ?", (person_id,)
            ).fetchall()
        with self._transaction() as connection:
            cursor = connection.execute("DELETE FROM people WHERE id = ?", (person_id,))
            if cursor.rowcount != 1:
                raise KeyError(person_id)
        person_dir = (self.photos / person_id).resolve()
        for row in rows:
            image_path = Path(row["image_path"]).resolve()
            if image_path.is_relative_to(person_dir):
                image_path.unlink(missing_ok=True)
        try:
            person_dir.rmdir()
        except OSError:
            pass

    def delete_face_sample(self, sample_id: str) -> None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT s.person_id, s.image_path,
                       (SELECT COUNT(*) FROM face_samples WHERE person_id = s.person_id) AS photo_count
                FROM face_samples s WHERE s.id = ?
                """,
                (sample_id,),
            ).fetchone()
        if row is None:
            raise KeyError(sample_id)
        if int(row["photo_count"]) <= 1:
            raise ValueError("每个人必须至少保留一张照片")
        with self._transaction() as connection:
            cursor = connection.execute("DELETE FROM face_samples WHERE id = ?", (sample_id,))
            if cursor.rowcount != 1:
                raise KeyError(sample_id)
        person_dir = (self.photos / row["person_id"]).resolve()
        image_path = Path(row["image_path"]).resolve()
        if image_path.is_relative_to(person_dir):
            image_path.unlink(missing_ok=True)

    def gallery(self, model_id: str) -> list[GallerySample]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT p.id AS person_id, p.name, p.id_card,
                       s.embedding, s.embedding_dim, s.quality
                FROM face_samples s JOIN people p ON p.id = s.person_id
                WHERE s.embedding_model = ?
                ORDER BY p.id, s.created_at
                """,
                (model_id,),
            ).fetchall()
        samples: list[GallerySample] = []
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if embedding.size != row["embedding_dim"]:
                continue
            norm = float(np.linalg.norm(embedding))
            if norm <= 1e-12:
                continue
            samples.append(GallerySample(
                person_id=row["person_id"],
                name=row["name"],
                id_card=row["id_card"],
                embedding=embedding / norm,
                quality=float(row["quality"]),
            ))
        return samples


def _validate_person(name: str, id_card: str) -> tuple[str, str]:
    name = name.strip()
    id_card = id_card.strip()
    if not name:
        raise ValueError("name is required")
    if not id_card:
        raise ValueError("ID card number is required")
    if len(name) > 100 or len(id_card) > 64:
        raise ValueError("person field is too long")
    return name, id_card
