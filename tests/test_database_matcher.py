from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from face_matching.database import EnrollmentSample, FaceDatabase
from face_matching.matcher import GalleryMatcher


def _sample(path, values, quality: float = 0.9) -> EnrollmentSample:
    path.write_bytes(b"photo")
    return EnrollmentSample(path, np.asarray(values, dtype=np.float32), quality)


def test_database_crud_and_gallery(tmp_path) -> None:
    database = FaceDatabase(tmp_path / "faces.sqlite3", tmp_path / "photos")
    first = _sample(tmp_path / "first.jpg", [1.0, 0.0, 0.0])
    second = _sample(tmp_path / "second.jpg", [0.8, 0.2, 0.0])

    person_id = database.add_person(" 张三 ", " 110101199001011234 ", [first], "model-v1")
    database.update_person(person_id, "张三", "110101199001011234", [second], "model-v1")

    person = database.get_person(person_id)
    assert person is not None
    assert (person.name, person.id_card, person.photo_count) == (
        "张三",
        "110101199001011234",
        2,
    )
    gallery = database.gallery("model-v1")
    assert len(gallery) == 2
    np.testing.assert_allclose(np.linalg.norm(gallery[1].embedding), 1.0, atol=1e-6)
    assert len(list((tmp_path / "photos" / person_id).iterdir())) == 2

    samples = database.list_face_samples(person_id)
    database.delete_face_sample(samples[0].id)
    assert database.get_person(person_id).photo_count == 1
    with pytest.raises(ValueError, match="至少保留"):
        database.delete_face_sample(database.list_face_samples(person_id)[0].id)

    with pytest.raises(sqlite3.IntegrityError):
        database.add_person("重复", "110101199001011234", [first], "model-v1")

    database.delete_person(person_id)
    assert database.list_people() == []
    assert not (tmp_path / "photos" / person_id).exists()
    database.close()


def test_matcher_uses_person_level_multi_photo_score_and_margin(tmp_path) -> None:
    database = FaceDatabase(tmp_path / "faces.sqlite3", tmp_path / "photos")
    alice = [
        _sample(tmp_path / "alice1.jpg", [1.0, 0.0], 1.0),
        _sample(tmp_path / "alice2.jpg", [0.95, 0.05], 0.8),
    ]
    bob = [_sample(tmp_path / "bob.jpg", [0.0, 1.0], 1.0)]
    alice_id = database.add_person("Alice", "A-1", alice, "model-v1")
    database.add_person("Bob", "B-1", bob, "model-v1")
    matcher = GalleryMatcher(database, "model-v1")

    accepted = matcher.match(np.array([1.0, 0.0]), threshold=0.5, min_margin=0.1)
    assert accepted.accepted
    assert accepted.person_id == alice_id
    assert accepted.name == "Alice"
    assert accepted.id_card == "A-1"

    ambiguous = matcher.match(np.array([1.0, 1.0]), threshold=0.5, min_margin=0.2)
    assert not ambiguous.accepted
    assert ambiguous.person_id is None
    database.close()
