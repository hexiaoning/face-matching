from __future__ import annotations

import shutil
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from face_matching.security import LocalVault, mask_government_id


@dataclass(frozen=True, slots=True)
class PersonSummary:
    id: int
    name: str
    masked_government_id: str
    photo_count: int


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    person_id: int
    name: str
    embedding: np.ndarray
    quality: float


class PeopleDatabase:
    def __init__(self, database_path: Path, enrollment_dir: Path, vault: LocalVault):
        self.database_path = database_path
        self.enrollment_dir = enrollment_dir
        self.vault = vault
        self._lock = threading.RLock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.enrollment_dir.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    government_id_cipher BLOB NOT NULL,
                    government_id_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    quality REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_photos_person ON photos(person_id);
                CREATE INDEX IF NOT EXISTS idx_photos_model ON photos(model_id);
                """
            )
            connection.commit()

    def add_person(
        self,
        name: str,
        government_id: str,
        photos: Sequence[tuple[Path, np.ndarray, float]],
        model_id: str,
    ) -> int:
        clean_name = name.strip()
        clean_id = "".join(government_id.split()).upper()
        if not clean_name:
            raise ValueError("姓名不能为空")
        if not clean_id:
            raise ValueError("身份证号不能为空")
        if not photos:
            raise ValueError("至少需要一张有效人脸照片")
        storage_id = uuid.uuid4().hex
        target_dir = self.enrollment_dir / storage_id
        target_dir.mkdir(parents=False, exist_ok=False)
        now = datetime.now(UTC).isoformat()
        copied: list[tuple[Path, np.ndarray, float]] = []
        try:
            for index, (source, embedding, quality) in enumerate(photos, start=1):
                suffix = source.suffix.lower() if source.suffix else ".jpg"
                target = target_dir / f"{index:03d}{suffix}"
                shutil.copy2(source, target)
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                copied.append((target, vector, float(quality)))
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO people(name, government_id_cipher, government_id_hash, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        clean_name,
                        self.vault.encrypt_text(clean_id),
                        self.vault.keyed_digest(clean_id),
                        now,
                    ),
                )
                person_id = int(cursor.lastrowid)
                connection.executemany(
                    """
                    INSERT INTO photos(
                        person_id, path, embedding, embedding_dim, model_id, quality, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            person_id,
                            str(path),
                            vector.tobytes(),
                            int(vector.size),
                            model_id,
                            quality,
                            now,
                        )
                        for path, vector, quality in copied
                    ],
                )
                connection.commit()
                return person_id
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

    def list_people(self) -> list[PersonSummary]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.id, p.name, p.government_id_cipher, COUNT(ph.id) AS photo_count
                FROM people p LEFT JOIN photos ph ON ph.person_id = p.id
                GROUP BY p.id ORDER BY p.name COLLATE NOCASE, p.id
                """
            ).fetchall()
        return [
            PersonSummary(
                id=int(row["id"]),
                name=str(row["name"]),
                masked_government_id=mask_government_id(
                    self.vault.decrypt_text(row["government_id_cipher"])
                ),
                photo_count=int(row["photo_count"]),
            )
            for row in rows
        ]

    def load_gallery(self, model_id: str) -> list[GalleryEntry]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.id AS person_id, p.name, ph.embedding, ph.embedding_dim, ph.quality
                FROM photos ph JOIN people p ON p.id = ph.person_id
                WHERE ph.model_id = ? ORDER BY p.id, ph.id
                """,
                (model_id,),
            ).fetchall()
        result: list[GalleryEntry] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if vector.size != int(row["embedding_dim"]):
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8:
                continue
            result.append(
                GalleryEntry(
                    person_id=int(row["person_id"]),
                    name=str(row["name"]),
                    embedding=vector / norm,
                    quality=float(row["quality"]),
                )
            )
        return result

    def delete_person(self, person_id: int) -> None:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT path FROM photos WHERE person_id = ?", (person_id,)
            ).fetchall()
            connection.execute("DELETE FROM people WHERE id = ?", (person_id,))
            connection.commit()
        roots = {Path(row["path"]).resolve().parent for row in rows}
        enrollment_root = self.enrollment_dir.resolve()
        for root in roots:
            if root != enrollment_root and root.is_relative_to(enrollment_root):
                shutil.rmtree(root, ignore_errors=True)

    def count_people(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM people").fetchone()
        return int(row["total"])
