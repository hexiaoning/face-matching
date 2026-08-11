import numpy as np

from face_matching.domain import FaceObservation, Person, PhotoRecord
from face_matching.gallery import GalleryIndex
from face_matching.tracking import MultiFaceTracker


def observation(x: float, embedding: list[float] | None) -> FaceObservation:
    return FaceObservation(
        bbox=np.asarray([x, 0, x + 100, 100], dtype=np.float32),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        detection_score=0.9,
        quality=0.8,
        embedding=np.asarray(embedding, dtype=np.float32) if embedding else None,
    )


def gallery() -> GalleryIndex:
    person = Person(7, "测试人员", "id-7", "now", "now", 1)
    photo = PhotoRecord(
        9,
        7,
        "face.jpg",
        "face.jpg",
        0.9,
        np.asarray([1, 0, 0], dtype=np.float32),
        "now",
    )
    result = GalleryIndex()
    result.rebuild([(photo, person)])
    return result


def test_track_requires_consistent_hits_before_confirmation() -> None:
    tracker = MultiFaceTracker(max_embeddings=4, ttl_frames=3)
    first_views, first_events = tracker.update(
        [observation(0, [1, 0, 0])], gallery(), 1, 0.5, 0.2, 2
    )
    assert not first_views[0].confirmed
    assert not first_events
    second_views, second_events = tracker.update(
        [observation(2, [1, 0, 0])], gallery(), 2, 0.5, 0.2, 2
    )
    assert second_views[0].confirmed
    assert len(second_events) == 1
    _, repeated_events = tracker.update(
        [observation(4, [1, 0, 0])], gallery(), 3, 0.5, 0.2, 2
    )
    assert not repeated_events


def test_iou_tracking_keeps_two_faces_separate() -> None:
    tracker = MultiFaceTracker()
    views, _ = tracker.update(
        [observation(0, None), observation(300, None)], GalleryIndex(), 1, 0.5, 0.2, 2
    )
    assert {view.track_id for view in views} == {1, 2}
    views, _ = tracker.update(
        [observation(305, None), observation(5, None)], GalleryIndex(), 2, 0.5, 0.2, 2
    )
    assert {view.track_id for view in views} == {1, 2}

