from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face_match.database import FaceDatabase, NewPhoto


def photo(path: Path, vector: list[float], quality: float = 0.8) -> NewPhoto:
    return NewPhoto(path, path.name, np.array(vector, dtype=np.float32), quality, "model-v1")


def test_person_and_multiple_photo_lifecycle(tmp_path: Path) -> None:
    database = FaceDatabase(tmp_path / "db.sqlite3")
    first = tmp_path / "one.jpg"
    second = tmp_path / "two.jpg"
    person_id = database.add_person(
        "张三", "passport-1", [photo(first, [1, 0, 0]), photo(second, [0.9, 0.1, 0])]
    )

    people = database.list_persons()
    assert [(item.id, item.name, item.photo_count) for item in people] == [(person_id, "张三", 2)]
    records = database.load_embeddings("model-v1")
    assert len(records) == 2
    assert all(np.linalg.norm(record.embedding) == pytest.approx(1.0) for record in records)

    third = tmp_path / "three.png"
    removed = database.update_person(
        person_id,
        "张三丰",
        "passport-2",
        [photo(third, [0, 1, 0])],
        [database.list_photos(person_id)[0].id],
    )
    assert removed == [first]
    assert database.get_person(person_id).name == "张三丰"
    assert len(database.list_photos(person_id)) == 2

    paths = database.delete_person(person_id)
    assert set(paths) == {second, third}
    assert database.list_persons() == []


def test_cannot_remove_last_photo(tmp_path: Path) -> None:
    database = FaceDatabase(tmp_path / "db.sqlite3")
    person_id = database.add_person("李四", "doc-2", [photo(tmp_path / "one.jpg", [1, 0])])
    only_photo = database.list_photos(person_id)[0]

    with pytest.raises(ValueError, match="至少需要保留"):
        database.update_person(person_id, "李四", "doc-2", [], [only_photo.id])

    assert database.get_person(person_id).photo_count == 1
