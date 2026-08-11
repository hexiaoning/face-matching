import sqlite3

import numpy as np
import pytest

from face_matching.database import FaceDatabase, FaceSampleInput
from face_matching.matching import GalleryMatcher
from face_matching.models import feature_model_id


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def sample(path: str, embedding: np.ndarray, model: str = "model-a", quality: float = 0.8):
    return FaceSampleInput(path, embedding, model, quality)


def test_database_crud_gallery_and_unique_id_card(tmp_path):
    database = FaceDatabase(tmp_path / "faces.db")
    alice_id = database.add_person(
        "Alice", "ID-001", [sample("alice-1.jpg", unit(1, 0, 0))]
    )
    database.add_samples(alice_id, [sample("alice-2.jpg", unit(0.98, 0.02, 0), quality=0.6)])

    record = database.get_person(alice_id)
    assert record is not None
    assert record.photo_count == 2
    assert database.list_gallery("other-model") == []
    assert len(database.list_gallery("model-a")) == 2

    database.update_person(alice_id, "Alice Zhang", "ID-001-A")
    assert database.get_person(alice_id).name == "Alice Zhang"
    bob_id = database.add_person(
        "Bob", "ID-002", [sample("bob.jpg", unit(0, 1, 0), model="model-b")]
    )
    with pytest.raises(ValueError, match="已经存在"):
        database.update_person(bob_id, "Bob", "ID-001-A")
    with pytest.raises(sqlite3.IntegrityError):
        database.add_person("Duplicate", "ID-001-A", [sample("dup.jpg", unit(0, 1, 0))])

    paths = database.delete_person(alice_id)
    assert set(paths) == {"alice-1.jpg", "alice-2.jpg"}
    assert database.get_person(alice_id) is None
    assert database.list_gallery("model-a") == []


def test_matcher_uses_multi_photo_templates_and_open_set_margin(tmp_path):
    database = FaceDatabase(tmp_path / "faces.db")
    alice = database.add_person(
        "Alice",
        "A",
        [
            sample("a1", unit(1.0, 0.0, 0.0), quality=1.0),
            sample("a2", unit(0.9, 0.1, 0.0), quality=0.7),
        ],
    )
    database.add_person("Bob", "B", [sample("b1", unit(0.0, 1.0, 0.0))])
    matcher = GalleryMatcher(database, "model-a", threshold=0.55, min_margin=0.08)

    accepted = matcher.match(unit(0.98, 0.02, 0.0))
    assert accepted.accepted is True
    assert accepted.person_id == alice
    assert accepted.name == "Alice"
    assert accepted.margin > 0.08

    ambiguous = matcher.match(unit(0.7, 0.7, 0.0))
    assert ambiguous.accepted is False
    assert ambiguous.person_id is None
    assert ambiguous.name == "陌生人"


def test_gallery_never_mixes_tta_and_single_embeddings(tmp_path):
    database = FaceDatabase(tmp_path / "faces.db")
    tta_model = feature_model_id("lvface-b", True)
    single_model = feature_model_id("lvface-b", False)
    database.add_person(
        "Alice", "A", [sample("a", unit(1, 0), model=tta_model)]
    )

    assert GalleryMatcher(database, tta_model).person_count == 1
    assert GalleryMatcher(database, single_model).person_count == 0


def test_corrupt_embedding_row_is_not_loaded(tmp_path):
    database = FaceDatabase(tmp_path / "faces.db")
    person_id = database.add_person("Alice", "A", [sample("a", unit(1, 0))])
    with database._connect() as connection:
        connection.execute(
            "UPDATE face_samples SET embedding_dim=999 WHERE person_id=?", (person_id,)
        )
    assert database.list_gallery("model-a") == []


def test_internal_event_log_preserves_video_source_verbatim(tmp_path):
    database = FaceDatabase(tmp_path / "faces.db")
    person_id = database.add_person("Alice", "110101199001010000", [sample("a", unit(1, 0))])
    source = "rtsp://admin:secret@192.0.2.1/live"

    database.log_event(person_id, source, track_id=7, score=0.81, quality=0.76)

    with database._connect() as connection:
        stored = connection.execute("SELECT source FROM recognition_events").fetchone()["source"]
    assert stored == source


def test_lvface_gallery_recalls_repeated_difficult_track_without_changing_gallery_flow(tmp_path):
    model_id = "lvface-b-glint360k-v1-tta"
    database = FaceDatabase(tmp_path / "faces.db")
    alice = database.add_person(
        "Alice", "A", [sample("a", unit(1.0, 0.0, 0.0), model=model_id)]
    )
    database.add_person(
        "Bob", "B", [sample("b", unit(0.0, 1.0, 0.0), model=model_id)]
    )
    matcher = GalleryMatcher(database, model_id, threshold=0.45, min_margin=0.06)
    difficult = unit(0.20, 0.0, np.sqrt(1.0 - 0.20**2))
    observations = [((difficult,), quality) for quality in (0.5, 0.6, 0.7, 0.8)]

    result = matcher.match_track(difficult, observations)

    assert result.accepted is True
    assert result.person_id == alice
    assert result.name == "Alice"
    assert result.score == pytest.approx(0.20)


def test_lvface_difficult_track_requires_independent_frames_and_identity_margin(tmp_path):
    model_id = "lvface-b-glint360k-v1-tta"
    database = FaceDatabase(tmp_path / "faces.db")
    database.add_person(
        "Alice", "A", [sample("a", unit(1.0, 0.0, 0.0), model=model_id)]
    )
    database.add_person(
        "Bob", "B", [sample("b", unit(0.0, 1.0, 0.0), model=model_id)]
    )
    matcher = GalleryMatcher(database, model_id, threshold=0.45, min_margin=0.06)
    strong = unit(0.20, 0.0, np.sqrt(1.0 - 0.20**2))
    one_frame = matcher.match_track(strong, [((strong, strong, strong), 0.8)])
    ambiguous = unit(0.20, 0.18, np.sqrt(1.0 - 0.20**2 - 0.18**2))
    repeated_ambiguous = matcher.match_track(
        ambiguous,
        [((ambiguous,), 0.8)] * 5,
    )

    assert one_frame.accepted is False
    assert repeated_ambiguous.accepted is False
