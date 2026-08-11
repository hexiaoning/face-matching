"""人员库：SQLite 存人员信息 + 照片 + 缓存的 embedding。"""
from __future__ import annotations

import shutil
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    id_number TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    embedding BLOB,
    quality REAL DEFAULT 0,
    UNIQUE(person_id, path)
);
CREATE INDEX IF NOT EXISTS idx_photos_person ON photos(person_id);
"""


@dataclass
class Photo:
    id: int
    path: str
    embedding: np.ndarray | None
    quality: float


@dataclass
class Person:
    id: int
    name: str
    id_number: str
    created_at: str
    photos: list[Photo] = field(default_factory=list)


class PersonDB:
    def __init__(self, db_file: Path | None = None):
        self.db_file = Path(db_file) if db_file else config.db_path()
        self.conn = sqlite3.connect(str(self.db_file))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- 人员 ----
    def add_person(self, name: str, id_number: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO persons(name, id_number) VALUES(?, ?)", (name.strip(), id_number.strip()))
        self.conn.commit()
        return int(cur.lastrowid)

    def update_person(self, person_id: int, name: str, id_number: str) -> None:
        self.conn.execute("UPDATE persons SET name=?, id_number=? WHERE id=?",
                          (name.strip(), id_number.strip(), person_id))
        self.conn.commit()

    def delete_person(self, person_id: int) -> None:
        paths = [r[0] for r in self.conn.execute(
            "SELECT path FROM photos WHERE person_id=?", (person_id,))]
        self.conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
        self.conn.commit()
        for p in paths:
            Path(p).unlink(missing_ok=True)

    def list_persons(self) -> list[Person]:
        out: list[Person] = []
        rows = self.conn.execute(
            "SELECT id, name, id_number, created_at FROM persons ORDER BY id").fetchall()
        for pid, name, idn, cat in rows:
            photos = [
                Photo(id=ph_id, path=ph_path,
                      embedding=(np.frombuffer(blob, dtype=np.float32).copy() if blob else None),
                      quality=qual)
                for ph_id, ph_path, blob, qual in self.conn.execute(
                    "SELECT id, path, embedding, quality FROM photos WHERE person_id=? ORDER BY id",
                    (pid,))
            ]
            out.append(Person(id=pid, name=name, id_number=idn, created_at=cat, photos=photos))
        return out

    # ---- 照片 ----
    def add_photo(self, person_id: int, src_path: str) -> int:
        """把照片拷入数据目录并登记；embedding 由调用方随后写入。"""
        src = Path(src_path)
        dst = config.photos_dir() / f"{person_id}_{uuid.uuid4().hex[:8]}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        cur = self.conn.execute(
            "INSERT INTO photos(person_id, path) VALUES(?, ?)", (person_id, str(dst)))
        self.conn.commit()
        return int(cur.lastrowid)

    def set_photo_embedding(self, photo_id: int, embedding: np.ndarray, quality: float) -> None:
        emb = np.ascontiguousarray(embedding, dtype=np.float32)
        self.conn.execute("UPDATE photos SET embedding=?, quality=? WHERE id=?",
                          (emb.tobytes(), float(quality), photo_id))
        self.conn.commit()

    def delete_photo(self, photo_id: int) -> None:
        row = self.conn.execute("SELECT path FROM photos WHERE id=?", (photo_id,)).fetchone()
        self.conn.execute("DELETE FROM photos WHERE id=?", (photo_id,))
        self.conn.commit()
        if row:
            Path(row[0]).unlink(missing_ok=True)

    # ---- 检索 ----
    def gallery(self) -> list[tuple[Person, np.ndarray]]:
        """返回 [(person, embedding)]，每人每张照片一条。"""
        out: list[tuple[Person, np.ndarray]] = []
        for person in self.list_persons():
            for ph in person.photos:
                if ph.embedding is not None:
                    out.append((person, ph.embedding))
        return out
