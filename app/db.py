"""SQLite 人员数据库：姓名 + 身份证号 + 1~N 张照片及其特征。"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    id_card    TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS photos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    embedding  BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS match_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER,
    name       TEXT,
    id_card    TEXT,
    score      REAL,
    source     TEXT,
    snapshot   TEXT,
    ts         REAL NOT NULL
);
"""


class Database:
    def __init__(self, path: Path | None = None):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path or config.DB_PATH), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- 人员管理 ----

    def add_person(self, name: str, id_card: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO persons(name, id_card, created_at) VALUES (?,?,?)",
            (name.strip(), id_card.strip(), time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def add_photo(self, person_id: int, path: str, embedding: np.ndarray) -> None:
        self._conn.execute(
            "INSERT INTO photos(person_id, path, embedding) VALUES (?,?,?)",
            (person_id, path, embedding.astype(np.float32).tobytes()),
        )
        self._conn.commit()

    def delete_person(self, person_id: int) -> list[str]:
        """删除人员及其照片记录，返回照片文件路径（由调用方清理文件）。"""
        rows = self._conn.execute(
            "SELECT path FROM photos WHERE person_id=?", (person_id,)
        ).fetchall()
        self._conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
        self._conn.commit()
        return [r[0] for r in rows]

    def list_persons(self) -> list[dict]:
        rows = self._conn.execute(
            """SELECT p.id, p.name, p.id_card, COUNT(ph.id)
               FROM persons p LEFT JOIN photos ph ON ph.person_id = p.id
               GROUP BY p.id ORDER BY p.id"""
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "id_card": r[2], "photo_count": r[3]}
            for r in rows
        ]

    def person_photos(self, person_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, path FROM photos WHERE person_id=?", (person_id,)
        ).fetchall()
        return [{"id": r[0], "path": r[1]} for r in rows]

    # ---- 特征库 ----

    def load_gallery(self) -> tuple[np.ndarray, list[dict]]:
        """返回 (N×512 特征矩阵, N 条人员信息)。每个人可能有多条（多张照片）。"""
        rows = self._conn.execute(
            """SELECT ph.embedding, p.id, p.name, p.id_card
               FROM photos ph JOIN persons p ON p.id = ph.person_id"""
        ).fetchall()
        embs, persons = [], []
        for blob, pid, name, id_card in rows:
            embs.append(np.frombuffer(blob, dtype=np.float32))
            persons.append({"id": pid, "name": name, "id_card": id_card})
        if embs:
            return np.stack(embs), persons
        return np.zeros((0, 512), dtype=np.float32), []

    # ---- 命中记录 ----

    def log_match(
        self,
        person: dict | None,
        score: float,
        source: str,
        snapshot_path: str | None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO match_events(person_id, name, id_card, score, source, snapshot, ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                person["id"] if person else None,
                person["name"] if person else None,
                person["id_card"] if person else None,
                score,
                source,
                snapshot_path,
                time.time(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
