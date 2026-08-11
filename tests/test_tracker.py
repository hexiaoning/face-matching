from __future__ import annotations

import numpy as np

from face_matching.detector import FaceDetection
from face_matching.matcher import MatchResult
from face_matching.tracker import FaceTrack, MultiFaceTracker, robust_aggregate


LANDMARKS = np.array(
    [[4.0, 5.0], [8.0, 5.0], [6.0, 7.0], [4.5, 9.0], [7.5, 9.0]],
    dtype=np.float32,
)


def _detection(x: float) -> FaceDetection:
    return FaceDetection(
        np.array([x, 0.0, x + 12.0, 12.0], dtype=np.float32),
        LANDMARKS + np.array([x, 0.0], dtype=np.float32),
        0.9,
    )


def test_tracker_keeps_id_for_nearby_detection() -> None:
    tracker = MultiFaceTracker(max_age=2)
    first = tracker.update([_detection(0.0)])[0][0]
    second = tracker.update([_detection(2.0)])[0][0]

    assert first.id == second.id
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert tracker.tracks == {}


def test_track_aggregation_confirmation_and_invalidation() -> None:
    track = FaceTrack(1, np.array([0, 0, 10, 10]), LANDMARKS, max_embeddings=3)
    track.add_embedding(np.array([1.0, 0.0]), 1.0, 1)
    track.add_embedding(np.array([0.8, 0.2]), 0.5, 2)
    aggregate = track.aggregate_embedding()
    assert aggregate is not None
    np.testing.assert_allclose(np.linalg.norm(aggregate), 1.0, atol=1e-6)
    assert aggregate[0] > aggregate[1]

    match = MatchResult("person", "张三", "123456789", 0.8, 0.3, True)
    assert not track.update_identity(match, confirmations=2)
    assert track.update_identity(match, confirmations=2)
    assert track.stable_match is not None

    track.invalidate_identity()
    assert track.stable_match is None
    assert track.last_match is None
    assert track.matched_embedding_version == -1


def test_robust_aggregation_rejects_identity_outlier() -> None:
    aggregate = robust_aggregate(
        [
            np.array([1.0, 0.0]),
            np.array([0.98, 0.02]),
            np.array([-1.0, 0.0]),
        ],
        [0.9, 0.8, 1.0],
    )
    assert aggregate is not None
    assert aggregate[0] > 0.99
    assert aggregate[1] > 0.0
