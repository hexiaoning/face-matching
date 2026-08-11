from __future__ import annotations

import numpy as np

from face_match.domain import EmbeddingRecord
from face_match.vision.matcher import MultiTemplateMatcher


def record(photo_id: int, person_id: int, name: str, vector: list[float]) -> EmbeddingRecord:
    value = np.array(vector, dtype=np.float32)
    value /= np.linalg.norm(value)
    return EmbeddingRecord(photo_id, person_id, name, f"id-{person_id}", value, 0.9)


def test_multi_photo_match_and_threshold() -> None:
    matcher = MultiTemplateMatcher()
    matcher.refresh(
        [
            record(1, 10, "Alice", [1, 0, 0]),
            record(2, 10, "Alice", [0.98, 0.2, 0]),
            record(3, 20, "Bob", [0, 1, 0]),
        ]
    )
    result = matcher.match(np.array([0.99, 0.05, 0]), threshold=0.5, ambiguity_margin=0.04)
    assert result.accepted
    assert result.person_id == 10
    assert result.name == "Alice"

    unknown = matcher.match(np.array([0, 0, 1]), threshold=0.5, ambiguity_margin=0.04)
    assert not unknown.accepted
    assert unknown.reason == "低于相似度阈值"


def test_ambiguous_top_two_is_rejected() -> None:
    matcher = MultiTemplateMatcher()
    matcher.refresh([record(1, 1, "A", [1, 0]), record(2, 2, "B", [0.99, 0.01])])
    result = matcher.match(np.array([1, 0]), threshold=0.4, ambiguity_margin=0.03)
    assert not result.accepted
    assert result.reason == "前两名过于接近"
