import numpy as np
import pytest

from face_matching.domain import Person, PhotoRecord
from face_matching.gallery import GalleryIndex


def person(person_id: int, name: str) -> Person:
    return Person(person_id, name, f"id-{person_id}", "now", "now", 2)


def photo(photo_id: int, person_id: int, vector: list[float]) -> PhotoRecord:
    return PhotoRecord(
        id=photo_id,
        person_id=person_id,
        path=f"{photo_id}.jpg",
        source_name=f"{photo_id}.jpg",
        quality=0.8,
        embedding=np.asarray(vector, dtype=np.float32),
        created_at="now",
    )


def test_multi_template_gallery_returns_best_identity() -> None:
    first = person(1, "甲")
    second = person(2, "乙")
    index = GalleryIndex()
    index.rebuild(
        [
            (photo(1, 1, [1, 0, 0]), first),
            (photo(2, 1, [0.9, 0.1, 0]), first),
            (photo(3, 2, [0, 1, 0]), second),
        ]
    )
    match = index.match(np.asarray([0.98, 0.02, 0], dtype=np.float32))
    assert match is not None
    assert match.person_id == 1
    assert match.score > 0.98
    assert index.person_count == 2
    assert index.photo_count == 3


def test_empty_gallery_has_no_match() -> None:
    assert GalleryIndex().match(np.asarray([1, 0], dtype=np.float32)) is None

