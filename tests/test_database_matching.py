import sqlite3

import numpy as np
import pytest

from face_matching.database import FaceDatabase, FaceSampleInput
from face_matching.matching import GalleryMatcher, TargetMatcher, TargetObservation
from face_matching.models import feature_model_id


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def target_observation(embeddings, quality=0.7, timestamp=0.0):
    values = embeddings if isinstance(embeddings, tuple) else (embeddings,)
    return TargetObservation(values, quality, timestamp)


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


def test_target_matcher_requires_multiple_video_frames_not_alignment_variants():
    reference = unit(1.0, 0.0, 0.0)
    matcher = TargetMatcher(
        [reference], threshold=0.75, review_threshold=0.40, min_support=2,
        auto_confirm=True,
    )
    strong = unit(0.9, 0.1, 0.0)
    alternate = unit(0.85, 0.15, 0.0)

    one_frame = matcher.match([target_observation((strong, alternate), 0.8, 0.0)])
    assert one_frame.decision == "review"
    assert one_frame.support == 1

    two_frames = matcher.match([
        target_observation((strong, alternate), 0.8, 0.0),
        target_observation(strong, 0.7, 1.0),
    ])
    assert two_frames.accepted is True
    assert two_frames.support == 2


def test_target_matcher_keeps_weak_but_relevant_track_for_review():
    matcher = TargetMatcher(
        [unit(1.0, 0.0)], threshold=0.80, review_threshold=0.30, min_support=2
    )

    result = matcher.match([target_observation(unit(0.5, 0.5), 0.6)])

    assert result.review is True
    assert result.accepted is False


def test_target_matcher_confirms_repeated_low_cross_quality_score():
    reference = unit(1.0, 0.0)
    difficult = unit(0.195, np.sqrt(1.0 - 0.195**2))
    matcher = TargetMatcher(
        [reference], threshold=0.19, review_threshold=0.12, min_support=2,
        auto_confirm=True,
    )

    first = matcher.match([target_observation(difficult, 0.7, 0.0)])
    confirmed = matcher.match([
        target_observation(difficult, 0.7, 0.0),
        target_observation(difficult, 0.6, 1.0),
    ])

    assert first.decision == "review"
    assert confirmed.accepted is True
    assert confirmed.support == 2
    assert confirmed.best_score == pytest.approx(0.195)


def test_target_matcher_does_not_confirm_one_spurious_high_frame():
    reference = unit(1.0, 0.0)
    high = unit(0.35, np.sqrt(1.0 - 0.35**2))
    low = unit(0.05, np.sqrt(1.0 - 0.05**2))
    matcher = TargetMatcher(
        [reference], threshold=0.19, review_threshold=0.12, min_support=2,
        auto_confirm=True,
    )

    result = matcher.match([
        target_observation(high, 0.8, 0.0),
        target_observation(low, 0.9, 1.0),
        target_observation(low, 0.7, 2.0),
    ])

    assert result.accepted is False
    assert result.decision == "review"
    assert result.support == 1


def test_target_matcher_does_not_count_adjacent_frames_as_independent_support():
    reference = unit(1.0, 0.0)
    difficult = unit(0.25, np.sqrt(1.0 - 0.25**2))
    matcher = TargetMatcher(
        [reference], threshold=0.19, review_threshold=0.12, min_support=2,
        min_evidence_gap=0.75, auto_confirm=True,
    )

    result = matcher.match([
        target_observation(difficult, 0.7, 0.0),
        target_observation(difficult, 0.8, 0.1),
        target_observation(difficult, 0.9, 0.2),
    ])

    assert result.decision == "review"
    assert result.support == 1
    assert result.evidence == 1


def test_target_matcher_keeps_auto_confirmation_disabled_until_calibrated():
    reference = unit(1.0, 0.0)
    matcher = TargetMatcher(
        [reference], threshold=0.19, review_threshold=0.12, min_support=2,
        auto_confirm=False,
    )

    result = matcher.match([
        target_observation(reference, 0.8, 0.0),
        target_observation(reference, 0.8, 1.0),
    ])

    assert result.decision == "review"
    assert result.accepted is False
