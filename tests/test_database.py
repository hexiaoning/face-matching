from pathlib import Path

import numpy as np
import pytest

from face_matching.database import Database, DuplicateIdNumberError
from face_matching.domain import PreparedPhoto


def prepared(name: str, vector: list[float], quality: float = 0.8) -> PreparedPhoto:
    return PreparedPhoto(
        path=name,
        source_name=name,
        quality=quality,
        embedding=np.asarray(vector, dtype=np.float32),
    )


def test_person_and_multiple_photos_round_trip(tmp_path: Path) -> None:
    database = Database(tmp_path / "gallery.sqlite3")
    person_id = database.add_person(
        "张三",
        "110101199001011234",
        [prepared("a.jpg", [3, 0, 0]), prepared("b.jpg", [2, 2, 0])],
    )
    people = database.list_persons("张")
    assert len(people) == 1
    assert people[0].id == person_id
    assert people[0].photo_count == 2
    photos = database.list_photos(person_id)
    assert len(photos) == 2
    assert np.linalg.norm(photos[0].embedding) == pytest.approx(1.0)
    database.close()


def test_id_number_is_unique(tmp_path: Path) -> None:
    database = Database(tmp_path / "gallery.sqlite3")
    database.add_person("甲", "same-id", [prepared("a.jpg", [1, 0])])
    with pytest.raises(DuplicateIdNumberError):
        database.add_person("乙", "same-id", [prepared("b.jpg", [0, 1])])
    database.close()


def test_update_cannot_remove_last_photo(tmp_path: Path) -> None:
    database = Database(tmp_path / "gallery.sqlite3")
    person_id = database.add_person("甲", "id-a", [prepared("a.jpg", [1, 0])])
    photo_id = database.list_photos(person_id)[0].id
    with pytest.raises(ValueError, match="至少"):
        database.update_person(person_id, "甲", "id-a", [], [photo_id])
    assert len(database.list_photos(person_id)) == 1
    database.close()

