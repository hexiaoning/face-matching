from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .paths import database_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class FaceSampleInput:
    image_path: str
    embedding: np.ndarray
    model_id: str
    quality: float


@dataclass(frozen=True, slots=True)
class PersonRecord:
    id: str
    name: str
    id_card: str
    photo_count: int
    created_at: str
    updated_at: str
    library_ids: tuple[str, ...] = ()
    library_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GallerySample:
    person_id: str
    name: str
    id_card: str
    embedding: np.ndarray
    quality: float
    library_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StoredSample:
    id: str
    person_id: str
    image_path: str
    model_id: str


class FaceDatabase:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    id_card TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS face_samples (
                    id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    image_path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    quality REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_samples_person ON face_samples(person_id);
                CREATE INDEX IF NOT EXISTS idx_samples_model ON face_samples(model_id);
                CREATE TABLE IF NOT EXISTS recognition_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT REFERENCES persons(id) ON DELETE SET NULL,
                    source TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    score REAL NOT NULL,
                    quality REAL NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recognition_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    frame_index INTEGER NOT NULL,
                    bbox TEXT,
                    snapshot_path TEXT,
                    face_path TEXT,
                    score REAL,
                    quality REAL,
                    model_id TEXT,
                    action TEXT NOT NULL CHECK(action IN (
                        'reject', 'associate', 'undo_rejection', 'remove_association'
                    )),
                    person_id TEXT REFERENCES persons(id) ON DELETE SET NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_annotations_session
                    ON recognition_annotations(session_id, track_id);
                CREATE TABLE IF NOT EXISTS portrait_libraries (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                );
                INSERT OR IGNORE INTO portrait_libraries VALUES('default', '默认库');
                CREATE TABLE IF NOT EXISTS person_libraries (
                    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    library_id TEXT NOT NULL REFERENCES portrait_libraries(id) ON DELETE CASCADE,
                    PRIMARY KEY(person_id, library_id)
                );
                """
            )
            if connection.execute("PRAGMA user_version").fetchone()[0] < 1:
                connection.execute("INSERT OR IGNORE INTO person_libraries SELECT id, 'default' FROM persons")
                connection.execute("PRAGMA user_version = 1")

    def add_person(
        self,
        name: str,
        id_card: str,
        samples: list[FaceSampleInput],
        person_id: str | None = None,
        library_ids: list[str] | None = None,
    ) -> str:
        name, id_card = name.strip(), id_card.strip()
        if not name or not id_card:
            raise ValueError("姓名和身份证号不能为空")
        if not samples:
            raise ValueError("至少需要一张有效照片")
        identifier = person_id or uuid.uuid4().hex
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO persons(id, name, id_card, created_at, updated_at) VALUES(?,?,?,?,?)",
                (identifier, name, id_card, timestamp, timestamp),
            )
            self._insert_samples(connection, identifier, samples, timestamp)
            connection.executemany("INSERT INTO person_libraries VALUES(?,?)",
                                   [(identifier, value) for value in dict.fromkeys(library_ids or ['default'])])
        return identifier

    @staticmethod
    def _insert_samples(
        connection: sqlite3.Connection,
        person_id: str,
        samples: list[FaceSampleInput],
        timestamp: str,
    ) -> None:
        rows = []
        for sample in samples:
            embedding = np.asarray(sample.embedding, dtype=np.float32).reshape(-1)
            if embedding.size == 0 or not np.isfinite(embedding).all():
                raise ValueError("人脸特征无效")
            rows.append(
                (
                    uuid.uuid4().hex,
                    person_id,
                    sample.image_path,
                    embedding.tobytes(),
                    int(embedding.size),
                    sample.model_id,
                    float(sample.quality),
                    timestamp,
                )
            )
        connection.executemany(
            """INSERT INTO face_samples(
                   id, person_id, image_path, embedding, embedding_dim, model_id, quality, created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            rows,
        )

    def add_samples(self, person_id: str, samples: list[FaceSampleInput]) -> None:
        if not samples:
            raise ValueError("没有可添加的照片")
        timestamp = _now()
        with self._connect() as connection:
            found = connection.execute("SELECT 1 FROM persons WHERE id=?", (person_id,)).fetchone()
            if not found:
                raise KeyError("人员不存在")
            self._insert_samples(connection, person_id, samples, timestamp)
            connection.execute("UPDATE persons SET updated_at=? WHERE id=?", (timestamp, person_id))

    def update_person(self, person_id: str, name: str, id_card: str) -> None:
        name, id_card = name.strip(), id_card.strip()
        if not name or not id_card:
            raise ValueError("姓名和身份证号不能为空")
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE persons SET name=?, id_card=?, updated_at=? WHERE id=?",
                    (name, id_card, _now(), person_id),
                )
                if cursor.rowcount == 0:
                    raise KeyError("人员不存在")
        except sqlite3.IntegrityError as exc:
            raise ValueError("该身份证号已经存在") from exc

    def get_person(self, person_id: str) -> PersonRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT p.*, COUNT(s.id) AS photo_count FROM persons p
                   LEFT JOIN face_samples s ON s.person_id=p.id
                   WHERE p.id=? GROUP BY p.id""",
                (person_id,),
            ).fetchone()
        return self._person_from_row(row) if row else None

    def _person_from_row(self, row: sqlite3.Row, libraries=None) -> PersonRecord:
        if libraries is None:
            with self._connect() as connection:
                libraries = connection.execute(
                    "SELECT l.id, l.name FROM portrait_libraries l JOIN person_libraries p ON l.id=p.library_id "
                    "WHERE p.person_id=? ORDER BY l.name", (row['id'],)
                ).fetchall()
        return PersonRecord(
            id=row["id"],
            name=row["name"],
            id_card=row["id_card"],
            photo_count=int(row["photo_count"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            library_ids=tuple(item['id'] for item in libraries),
            library_names=tuple(item['name'] for item in libraries),
        )

    def list_libraries(self) -> list[tuple[str, str]]:
        with self._connect() as connection:
            return [(row['id'], row['name']) for row in connection.execute(
                "SELECT id, name FROM portrait_libraries ORDER BY name")]

    def create_library(self, name: str) -> str:
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("库名称须为 1～100 个字符")
        identifier = uuid.uuid4().hex
        try:
            with self._connect() as connection:
                connection.execute("INSERT INTO portrait_libraries VALUES(?,?)", (identifier, name))
        except sqlite3.IntegrityError as exc:
            raise ValueError("该库名称已经存在") from exc
        return identifier

    def rename_library(self, library_id: str, name: str) -> None:
        name = name.strip()
        if library_id == 'default':
            raise ValueError("默认库不能重命名")
        if not name or len(name) > 100:
            raise ValueError("库名称须为 1～100 个字符")
        try:
            with self._connect() as connection:
                if not connection.execute("UPDATE portrait_libraries SET name=? WHERE id=?", (name, library_id)).rowcount:
                    raise KeyError("库不存在")
        except sqlite3.IntegrityError as exc:
            raise ValueError("该库名称已经存在") from exc

    def delete_library(self, library_id: str) -> None:
        if library_id == 'default':
            raise ValueError("默认库不能删除")
        with self._connect() as connection:
            connection.execute("DELETE FROM portrait_libraries WHERE id=?", (library_id,))
            connection.execute("INSERT OR IGNORE INTO person_libraries SELECT id, 'default' FROM persons "
                               "WHERE id NOT IN (SELECT person_id FROM person_libraries)")

    def set_person_libraries(self, person_id: str, library_ids: list[str]) -> None:
        with self._connect() as connection:
            if not connection.execute("SELECT 1 FROM persons WHERE id=?", (person_id,)).fetchone():
                raise KeyError("人员不存在")
            connection.execute("DELETE FROM person_libraries WHERE person_id=?", (person_id,))
            connection.executemany("INSERT INTO person_libraries VALUES(?,?)",
                                   [(person_id, value) for value in dict.fromkeys(library_ids or ['default'])])
            connection.execute("UPDATE persons SET updated_at=? WHERE id=?", (_now(), person_id))

    def list_people(self) -> list[PersonRecord]:
        with self._connect() as connection:
            libraries: dict[str, list] = {}
            for item in connection.execute(
                "SELECT p.person_id, l.id, l.name FROM person_libraries p JOIN portrait_libraries l "
                "ON l.id=p.library_id ORDER BY l.name"
            ):
                libraries.setdefault(item['person_id'], []).append(item)
            rows = connection.execute(
                """SELECT p.*, COUNT(s.id) AS photo_count FROM persons p
                   LEFT JOIN face_samples s ON s.person_id=p.id
                   GROUP BY p.id ORDER BY p.name COLLATE NOCASE, p.created_at"""
            ).fetchall()
        return [self._person_from_row(row, libraries.get(row['id'], [])) for row in rows]

    def delete_person(self, person_id: str) -> list[str]:
        with self._connect() as connection:
            paths = [
                row[0]
                for row in connection.execute(
                    "SELECT image_path FROM face_samples WHERE person_id=?", (person_id,)
                ).fetchall()
            ]
            cursor = connection.execute("DELETE FROM persons WHERE id=?", (person_id,))
            if cursor.rowcount == 0:
                raise KeyError("人员不存在")
        return paths

    def list_gallery(self, model_id: str) -> list[GallerySample]:
        with self._connect() as connection:
            memberships: dict[str, list[str]] = {}
            for member in connection.execute("SELECT person_id, library_id FROM person_libraries"):
                memberships.setdefault(member['person_id'], []).append(member['library_id'])
            rows = connection.execute(
                """SELECT p.id AS person_id, p.name, p.id_card,
                          s.embedding, s.embedding_dim, s.quality
                   FROM face_samples s JOIN persons p ON p.id=s.person_id
                   WHERE s.model_id=? ORDER BY p.id, s.quality DESC""",
                (model_id,),
            ).fetchall()
        result: list[GallerySample] = []
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if embedding.size != int(row["embedding_dim"]):
                continue
            result.append(
                GallerySample(
                    person_id=row["person_id"],
                    name=row["name"],
                    id_card=row["id_card"],
                    embedding=embedding,
                    quality=float(row["quality"]),
                    library_ids=tuple(memberships.get(row['person_id'], [])),
                )
            )
        return result

    def list_stored_samples(self) -> list[StoredSample]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, person_id, image_path, model_id FROM face_samples ORDER BY person_id, created_at"
            ).fetchall()
        return [StoredSample(row["id"], row["person_id"], row["image_path"], row["model_id"]) for row in rows]

    def update_sample_feature(
        self,
        sample_id: str,
        embedding: np.ndarray,
        model_id: str,
        quality: float,
    ) -> None:
        value = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError("人脸特征无效")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE face_samples SET embedding=?, embedding_dim=?, model_id=?, quality=?
                   WHERE id=?""",
                (value.tobytes(), int(value.size), model_id, float(quality), sample_id),
            )
            if cursor.rowcount == 0:
                raise KeyError("照片记录不存在")

    def log_event(
        self,
        person_id: str,
        source: str,
        track_id: int,
        score: float,
        quality: float,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO recognition_events(
                       person_id, source, track_id, score, quality, occurred_at
                   ) VALUES(?,?,?,?,?,?)""",
                (person_id, source, int(track_id), float(score), float(quality), _now()),
            )

    def log_annotation(
        self,
        session_id: str,
        source: str,
        track_id: int,
        action: str,
        person_id: str,
        frame_index: int = 0,
        bbox: tuple[float, float, float, float] | None = None,
        snapshot_path: str | None = None,
        face_path: str | None = None,
        *,
        score: float | None = None,
        quality: float | None = None,
        model_id: str | None = None,
        resolve_conflicts: bool = False,
    ) -> None:
        if action not in {"reject", "associate", "undo_rejection", "remove_association"}:
            raise ValueError("未知标注操作")
        with self._connect() as connection:
            actions = []
            if resolve_conflicts and action == "associate":
                previous = connection.execute(
                    """SELECT action FROM recognition_annotations
                       WHERE session_id=? AND track_id=? AND person_id=?
                       AND action IN ('reject', 'undo_rejection') ORDER BY id DESC LIMIT 1""",
                    (session_id, track_id, person_id),
                ).fetchone()
                if previous and previous["action"] == "reject":
                    actions.append("undo_rejection")
            elif resolve_conflicts and action == "reject":
                previous = connection.execute(
                    """SELECT action, person_id FROM recognition_annotations
                       WHERE session_id=? AND track_id=?
                       AND action IN ('associate', 'remove_association') ORDER BY id DESC LIMIT 1""",
                    (session_id, track_id),
                ).fetchone()
                if previous and previous["action"] == "associate" and previous["person_id"] == person_id:
                    actions.append("remove_association")
            actions.append(action)
            connection.executemany(
                """INSERT INTO recognition_annotations(
                       session_id, source, track_id, frame_index, bbox, snapshot_path, face_path,
                       score, quality, model_id, action,
                       person_id, occurred_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [(session_id, source, int(track_id), int(frame_index),
                 json.dumps(bbox) if bbox is not None else None,
                 snapshot_path, face_path, score, quality, model_id, operation, person_id, _now())
                 for operation in actions],
            )

    def list_annotations(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM recognition_annotations ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]
