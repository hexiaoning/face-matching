from __future__ import annotations

import numpy as np

from face_match.domain import FaceDetection, FaceObservation, FaceQuality
from face_match.vision.tracker import FaceTracker, intersection_over_union, robust_aggregate

QUALITY = FaceQuality(0.8, 0.8, 0.8, 0.8, 0.8)


def observation(box: list[float], embedding: list[float] | None) -> FaceObservation:
    detection = FaceDetection(
        np.array(box, dtype=np.float32), 0.9, np.zeros((5, 2), dtype=np.float32)
    )
    vector = None if embedding is None else np.array(embedding, dtype=np.float32)
    if vector is not None:
        vector /= np.linalg.norm(vector)
    return FaceObservation(detection, QUALITY, vector)


def test_track_survives_motion_and_aggregates() -> None:
    tracker = FaceTracker(maximum_samples=4)
    first = tracker.update([observation([0, 0, 100, 100], [1, 0, 0])])[0]
    second = tracker.update([observation([10, 5, 110, 105], [0.98, 0.1, 0])])[0]
    assert first.track_id == second.track_id
    assert len(second.embeddings) == 2
    assert second.aggregate[0] > 0.99


def test_robust_aggregation_suppresses_outlier() -> None:
    samples = [
        (0.9, np.array([1.0, 0.0, 0.0])),
        (0.8, np.array([0.99, 0.05, 0.0])),
        (0.7, np.array([0.0, 1.0, 0.0])),
    ]
    aggregate = robust_aggregate(samples)
    assert aggregate[0] > 0.99
    assert aggregate[1] < 0.1
    assert intersection_over_union(np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15])) > 0
