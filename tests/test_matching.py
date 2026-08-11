import numpy as np

from face_matching.database import GalleryEntry
from face_matching.inference import EmbeddedFace
from face_matching.matching import IdentityMatcher, TemporalTracker


def _entry(person_id, name, vector):
    value = np.asarray(vector, dtype=np.float32)
    value /= np.linalg.norm(value)
    return GalleryEntry(person_id, name, value, 1.0)


def test_multi_photo_identity_uses_best_template():
    matcher = IdentityMatcher(threshold=0.8, min_margin=0.05)
    matcher.replace_gallery(
        [
            _entry(1, "甲", [1, 0, 0]),
            _entry(1, "甲", [0, 1, 0]),
            _entry(2, "乙", [0, 0, 1]),
        ]
    )
    result = matcher.match(np.array([0.05, 0.99, 0], dtype=np.float32))
    assert result.accepted
    assert result.person_id == 1
    assert result.name == "甲"


def test_margin_rejects_ambiguous_probe():
    matcher = IdentityMatcher(threshold=0.6, min_margin=0.10)
    matcher.replace_gallery([_entry(1, "甲", [1, 0]), _entry(2, "乙", [0.98, 0.2])])
    result = matcher.match(np.array([1, 0], dtype=np.float32))
    assert not result.accepted
    assert result.name == "未知"


def test_empty_gallery_is_unknown():
    result = IdentityMatcher(0.4, 0.05).match(np.array([1, 0], dtype=np.float32))
    assert not result.accepted
    assert result.score == -1.0


def _face(vector, quality=0.9):
    embedding = np.asarray(vector, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)
    return EmbeddedFace(
        bbox=np.array([10, 10, 80, 80], dtype=np.float32),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        detection_score=0.99,
        embedding=embedding,
        quality=quality,
    )


def test_track_requires_consecutive_identity_confirmation():
    matcher = IdentityMatcher(threshold=0.7, min_margin=0.05)
    matcher.replace_gallery([_entry(1, "甲", [1, 0])])
    tracker = TemporalTracker(max_age=3, min_confirmations=3, min_quality=0.3)

    first = tracker.update([_face([1, 0])], 1, matcher)[0]
    first_accepted = first.match.accepted
    second = tracker.update([_face([1, 0])], 2, matcher)[0]
    second_accepted = second.match.accepted
    third = tracker.update([_face([1, 0])], 3, matcher)[0]

    assert not first_accepted
    assert not second_accepted
    assert third.match.accepted
    assert third.match.name == "甲"
    assert third.candidate_hits == 3


def test_low_quality_track_is_never_automatically_confirmed():
    matcher = IdentityMatcher(threshold=0.7, min_margin=0.05)
    matcher.replace_gallery([_entry(1, "甲", [1, 0])])
    tracker = TemporalTracker(max_age=3, min_confirmations=1, min_quality=0.5)

    track = tracker.update([_face([1, 0], quality=0.2)], 1, matcher)[0]

    assert not track.match.accepted
    assert track.decision == "画质不足"
    assert track.candidate_hits == 0
