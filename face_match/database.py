from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from face_match.domain import EmbeddingRecord, Person, PersonPhoto


@dataclass(frozen=True)
class NewPhoto:
    path: Path
    original_name: str
    embedding: np.ndarray
    quality: float
    model_version: str


class FaceDatabase:
    """Small local SQLite store. Every public operation is transaction-safe."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterable[sqlite3.Connection]:
        with self._lock, self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    id_number TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    file_path TEXT NOT NULL UNIQUE,
                    original_name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    quality REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_photos_person ON photos(person_id);
                CREATE INDEX IF NOT EXISTS idx_photos_model ON photos(model_version);
                PRAGMA user_version = 1;
                """
            )

    @staticmethod
    def _person_from_row(row: sqlite3.Row) -> Person:
        return Person(
            id=int(row["id"]),
            name=str(row["name"]),
            id_number=str(row["id_number"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            photo_count=int(row["photo_count"]),
        )

    def list_persons(self) -> list[Person]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*, COUNT(ph.id) AS photo_count
                FROM persons p
                LEFT JOIN photos ph ON ph.person_id = p.id
                GROUP BY p.id
                ORDER BY p.name COLLATE NOCASE, p.id
                """
            ).fetchall()
        return [self._person_from_row(row) for row in rows]

    def get_person(self, person_id: int) -> Person | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT p.*, COUNT(ph.id) AS photo_count
                FROM persons p
                LEFT JOIN photos ph ON ph.person_id = p.id
                WHERE p.id = ?
                GROUP BY p.id
                """,
                (person_id,),
            ).fetchone()
        return self._person_from_row(row) if row else None

    def list_photos(self, person_id: int) -> list[PersonPhoto]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, person_id, file_path, original_name, quality, model_version
                FROM photos WHERE person_id = ? ORDER BY id
                """,
                (person_id,),
            ).fetchall()
        return [
            PersonPhoto(
                id=int(row["id"]),
                person_id=int(row["person_id"]),
                path=Path(str(row["file_path"])),
                original_name=str(row["original_name"]),
                quality=float(row["quality"]),
                model_version=str(row["model_version"]),
            )
            for row in rows
        ]

    def add_person(self, name: str, id_number: str, photos: Sequence[NewPhoto]) -> int:
        if not photos:
            raise ValueError("每个人至少需要一张照片")
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO persons(name, id_number) VALUES (?, ?)", (name, id_number)
            )
            person_id = int(cursor.lastrowid)
            self._insert_photos(connection, person_id, photos)
        return person_id

    def update_person(
        self,
        person_id: int,
        name: str,
        id_number: str,
        new_photos: Sequence[NewPhoto],
        remove_photo_ids: Sequence[int],
    ) -> list[Path]:
        removed_paths: list[Path] = []
        with self._transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM persons WHERE id = ?", (person_id,)
            ).fetchone()
            if not exists:
                raise KeyError(f"人员不存在：{person_id}")
            if remove_photo_ids:
                placeholders = ",".join("?" for _ in remove_photo_ids)
                parameters = (person_id, *[int(item) for item in remove_photo_ids])
                rows = connection.execute(
                    f"SELECT file_path FROM photos WHERE person_id = ? AND id IN ({placeholders})",
                    parameters,
                ).fetchall()
                removed_paths = [Path(str(row["file_path"])) for row in rows]
                connection.execute(
                    f"DELETE FROM photos WHERE person_id = ? AND id IN ({placeholders})", parameters
                )
            remaining = int(
                connection.execute(
                    "SELECT COUNT(*) FROM photos WHERE person_id = ?", (person_id,)
                ).fetchone()[0]
            )
            if remaining + len(new_photos) < 1:
                raise ValueError("每个人至少需要保留一张照片")
            self._insert_photos(connection, person_id, new_photos)
            connection.execute(
                """
                UPDATE persons
                SET name = ?, id_number = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (name, id_number, person_id),
            )
        return removed_paths

    @staticmethod
    def _insert_photos(
        connection: sqlite3.Connection, person_id: int, photos: Sequence[NewPhoto]
    ) -> None:
        for photo in photos:
            vector = np.asarray(photo.embedding, dtype=np.float32).reshape(-1)
            connection.execute(
                """
                INSERT INTO photos(
                    person_id, file_path, original_name, embedding,
                    embedding_dim, quality, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    str(photo.path),
                    photo.original_name,
                    vector.tobytes(),
                    int(vector.size),
                    float(photo.quality),
                    photo.model_version,
                ),
            )

    def delete_person(self, person_id: int) -> list[Path]:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT file_path FROM photos WHERE person_id = ?", (person_id,)
            ).fetchall()
            paths = [Path(str(row["file_path"])) for row in rows]
            connection.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        return paths

    def load_embeddings(self, model_version: str) -> list[EmbeddingRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT ph.id AS photo_id, ph.person_id, p.name, p.id_number,
                       ph.embedding, ph.embedding_dim, ph.quality
                FROM photos ph
                JOIN persons p ON p.id = ph.person_id
                WHERE ph.model_version = ?
                ORDER BY ph.person_id, ph.id
                """,
                (model_version,),
            ).fetchall()
        records: list[EmbeddingRecord] = []
        for row in rows:
            dimension = int(row["embedding_dim"])
            vector = np.frombuffer(row["embedding"], dtype=np.float32, count=dimension).copy()
            norm = float(np.linalg.norm(vector))
            if vector.size == dimension and norm > 1e-8 and np.isfinite(vector).all():
                records.append(
                    EmbeddingRecord(
                        photo_id=int(row["photo_id"]),
                        person_id=int(row["person_id"]),
                        person_name=str(row["name"]),
                        id_number=str(row["id_number"]),
                        embedding=vector / norm,
                        quality=float(row["quality"]),
                    )
                )
        return records

    def count_embeddings(self, model_version: str) -> tuple[int, int]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(DISTINCT person_id), COUNT(*) FROM photos WHERE model_version = ?
                """,
                (model_version,),
            ).fetchone()
        return int(row[0]), int(row[1])
