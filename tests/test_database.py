import sqlite3
from pathlib import Path

import numpy as np
import pytest

from face_matching.database import PeopleDatabase
from face_matching.security import LocalVault


def _database(tmp_path: Path) -> PeopleDatabase:
    return PeopleDatabase(
        tmp_path / "people.sqlite3",
        tmp_path / "enrollment",
        LocalVault(tmp_path / "master.key"),
    )


def test_person_crud_and_gallery(tmp_path):
    source = tmp_path / "portrait.jpg"
    source.write_bytes(b"test-image")
    vector = np.array([3.0, 4.0], dtype=np.float32)
    database = _database(tmp_path)
    person_id = database.add_person(
        "张三", "110101199001011234", [(source, vector, 0.8)], "model-a"
    )

    people = database.list_people()
    assert [(item.id, item.name, item.photo_count) for item in people] == [(person_id, "张三", 1)]
    assert people[0].masked_government_id.endswith("1234")
    gallery = database.load_gallery("model-a")
    assert len(gallery) == 1
    np.testing.assert_allclose(gallery[0].embedding, [0.6, 0.8])
    assert database.load_gallery("another-model") == []

    copied_path = next((tmp_path / "enrollment").rglob("*.jpg"))
    assert copied_path.exists()
    database.delete_person(person_id)
    assert database.list_people() == []
    assert not copied_path.exists()


def test_duplicate_government_id_is_rejected_and_copy_is_cleaned(tmp_path):
    source = tmp_path / "portrait.jpg"
    source.write_bytes(b"test-image")
    database = _database(tmp_path)
    photo = (source, np.array([1.0, 0.0], dtype=np.float32), 0.9)
    database.add_person("甲", "ABC123", [photo], "model-a")
    before = {path for path in (tmp_path / "enrollment").iterdir()}
    with pytest.raises(sqlite3.IntegrityError):
        database.add_person("乙", " abc123 ", [photo], "model-a")
    after = {path for path in (tmp_path / "enrollment").iterdir()}
    assert after == before
