from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .domain import Person, PhotoRecord, PreparedPhoto, normalize_embedding


class DuplicateIdNumberError(ValueError):
    pass


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False, timeout=15)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _transaction(self):
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def _migrate(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL CHECK(length(trim(name)) > 0),
                    id_number TEXT NOT NULL UNIQUE CHECK(length(trim(id_number)) > 0),
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    path TEXT NOT NULL UNIQUE,
                    source_name TEXT NOT NULL,
                    quality REAL NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_photos_person_id ON photos(person_id);

                CREATE TABLE IF NOT EXISTS recognition_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    score REAL NOT NULL,
                    quality REAL NOT NULL,
                    occurred_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_events_occurred_at
                    ON recognition_events(occurred_at DESC);
                PRAGMA user_version = 1;
                """
            )
            self._connection.commit()

    @staticmethod
    def _person_from_row(row: sqlite3.Row) -> Person:
        return Person(
            id=row["id"],
            name=row["name"],
            id_number=row["id_number"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            photo_count=row["photo_count"] if "photo_count" in row.keys() else 0,
        )

    def list_persons(self, search: str = "") -> list[Person]:
        query = """
            SELECT p.*, COUNT(ph.id) AS photo_count
            FROM persons p LEFT JOIN photos ph ON ph.person_id = p.id
        """
        parameters: list[object] = []
        if search.strip():
            query += " WHERE p.name LIKE ? ESCAPE '\\' OR p.id_number LIKE ? ESCAPE '\\'"
            escaped = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            parameters.extend([f"%{escaped}%", f"%{escaped}%"])
        query += " GROUP BY p.id ORDER BY p.updated_at DESC, p.id DESC"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [self._person_from_row(row) for row in rows]

    def get_person(self, person_id: int) -> Person | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT p.*, COUNT(ph.id) AS photo_count
                FROM persons p LEFT JOIN photos ph ON ph.person_id = p.id
                WHERE p.id = ? GROUP BY p.id
                """,
                (person_id,),
            ).fetchone()
        return self._person_from_row(row) if row else None

    def add_person(self, name: str, id_number: str, photos: Iterable[PreparedPhoto]) -> int:
        prepared = list(photos)
        if not prepared:
            raise ValueError("每个人至少需要一张有效照片")
        try:
            with self._transaction() as connection:
                cursor = connection.execute(
                    "INSERT INTO persons(name, id_number) VALUES (?, ?)",
                    (name.strip(), id_number.strip()),
                )
                person_id = int(cursor.lastrowid)
                self._insert_photos(connection, person_id, prepared)
            return person_id
        except sqlite3.IntegrityError as exc:
            if "id_number" in str(exc).lower() or "unique" in str(exc).lower():
                raise DuplicateIdNumberError("该身份证号已存在") from exc
            raise

    def update_person(
        self,
        person_id: int,
        name: str,
        id_number: str,
        new_photos: Iterable[PreparedPhoto],
        removed_photo_ids: Iterable[int],
    ) -> list[str]:
        additions = list(new_photos)
        removals = [int(value) for value in removed_photo_ids]
        removed_paths: list[str] = []
        try:
            with self._transaction() as connection:
                existing = connection.execute(
                    "SELECT COUNT(*) FROM photos WHERE person_id = ?", (person_id,)
                ).fetchone()[0]
                owned_removals = []
                if removals:
                    placeholders = ",".join("?" for _ in removals)
                    rows = connection.execute(
                        f"SELECT id, path FROM photos WHERE person_id = ? AND id IN ({placeholders})",
                        [person_id, *removals],
                    ).fetchall()
                    owned_removals = [row["id"] for row in rows]
                    removed_paths = [row["path"] for row in rows]
                if existing - len(owned_removals) + len(additions) < 1:
                    raise ValueError("每个人至少需要保留一张有效照片")
                connection.execute(
                    """
                    UPDATE persons SET name = ?, id_number = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?
                    """,
                    (name.strip(), id_number.strip(), person_id),
                )
                if owned_removals:
                    placeholders = ",".join("?" for _ in owned_removals)
                    connection.execute(
                        f"DELETE FROM photos WHERE person_id = ? AND id IN ({placeholders})",
                        [person_id, *owned_removals],
                    )
                self._insert_photos(connection, person_id, additions)
            return removed_paths
        except sqlite3.IntegrityError as exc:
            if "id_number" in str(exc).lower() or "unique" in str(exc).lower():
                raise DuplicateIdNumberError("该身份证号已存在") from exc
            raise

    @staticmethod
    def _insert_photos(
        connection: sqlite3.Connection, person_id: int, photos: Iterable[PreparedPhoto]
    ) -> None:
        for photo in photos:
            vector = normalize_embedding(photo.embedding)
            connection.execute(
                """
                INSERT INTO photos(person_id, path, source_name, quality, embedding, embedding_dim)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    photo.path,
                    photo.source_name,
                    float(photo.quality),
                    vector.tobytes(),
                    int(vector.size),
                ),
            )

    def delete_person(self, person_id: int) -> list[str]:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT path FROM photos WHERE person_id = ?", (person_id,)
            ).fetchall()
            connection.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        return [row["path"] for row in rows]

    def list_photos(self, person_id: int | None = None) -> list[PhotoRecord]:
        query = "SELECT * FROM photos"
        parameters: tuple[object, ...] = ()
        if person_id is not None:
            query += " WHERE person_id = ?"
            parameters = (person_id,)
        query += " ORDER BY id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        photos: list[PhotoRecord] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if vector.size != row["embedding_dim"]:
                continue
            photos.append(
                PhotoRecord(
                    id=row["id"],
                    person_id=row["person_id"],
                    path=row["path"],
                    source_name=row["source_name"],
                    quality=row["quality"],
                    embedding=vector,
                    created_at=row["created_at"],
                )
            )
        return photos

    def gallery_rows(self) -> list[tuple[PhotoRecord, Person]]:
        persons = {person.id: person for person in self.list_persons()}
        return [
            (photo, persons[photo.person_id])
            for photo in self.list_photos()
            if photo.person_id in persons
        ]

    def log_recognition(
        self, person_id: int, source: str, track_id: int, score: float, quality: float
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO recognition_events(person_id, source, track_id, score, quality)
                VALUES (?, ?, ?, ?, ?)
                """,
                (person_id, source, track_id, float(score), float(quality)),
            )

