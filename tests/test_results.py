from face_matching.results import VideoMatch, VideoMatchList
from dataclasses import replace


def event(person_id: str, score: float, track_id: int = 1) -> dict[str, object]:
    return {
        "time": "12:00:00",
        "track_id": track_id,
        "person_id": person_id,
        "name": f"Person {person_id}",
        "id_card": f"ID-{person_id}",
        "score": score,
        "quality": 0.8,
    }


def test_video_match_list_keeps_each_person_best_score_and_sorts_descending():
    matches = VideoMatchList()
    matches.update(event("a", 0.71, 1))
    matches.update(event("b", 0.88, 2))
    assert matches.update(event("a", 0.69, 3)) is False
    assert matches.update(event("a", 0.91, 4)) is True

    ranked = matches.ranked()

    assert [item.person_id for item in ranked] == ["a", "b"]
    assert [item.score for item in ranked] == [0.91, 0.88]
    assert ranked[0].track_id == 4


def test_rejection_survives_new_scores_and_keeps_other_tracks_available():
    matches = VideoMatchList()
    matches.update(event("a", 0.9, 1))
    matches.update(event("a", 0.7, 2))
    matches.reject(1, "a")

    matches.update(event("a", 0.99, 1))
    assert matches.ranked()[0].track_id == 2
    assert matches.displayed()[-1].status == "不是此人"
    assert matches.displayed()[-1].score == 0.9

    matches.undo_rejection(1, "a")
    assert matches.ranked()[0].track_id == 1
    matches.clear()
    assert matches.displayed() == ()
    assert not matches.is_rejected(1, "a")


def test_manual_association_overrides_automatic_identity_until_removed():
    matches = VideoMatchList()
    matches.update(event("a", 0.9, 1))
    manual = VideoMatch("12:01:00", 1, "b", "Person b", "ID-b", 0.0, 0.8)
    matches.associate(manual)
    matches.update(event("a", 0.99, 1))
    matches.update(event("b", 0.95, 2))

    assert len(matches) == 1
    assert matches.ranked()[0].track_id == 1
    assert matches.ranked()[0].status == "人工关联"
    matches.remove_association(1)
    assert [match.person_id for match in matches.ranked()] == ["a", "b"]
    assert matches.ranked()[0].score == 0.99


def test_candidates_are_visible_but_do_not_count_as_confirmed_and_can_be_reviewed():
    matches = VideoMatchList()
    pending = {**event("a", 0.41), "status": "待确认", "reason": "低于阈值"}
    matches.update(pending)
    assert matches.displayed()[0].name == "Person a"
    assert matches.displayed()[0].reason == "低于阈值"
    assert len(matches) == 0 and matches.pending_count == 1
    matches.reject(1, "a")
    assert matches.pending_count == 0
    matches.undo_rejection(1, "a")
    assert matches.pending_count == 1


def test_confirmation_takes_precedence_over_a_higher_pending_score():
    matches = VideoMatchList()
    matches.update({**event("a", 0.9), "status": "待确认"})
    assert matches.update(event("a", 0.7))
    assert matches.displayed()[0].status == "自动匹配"
    assert matches.displayed()[0].score == 0.7
    assert len(matches) == 1 and matches.pending_count == 0
    assert not matches.update({**event("a", 0.99), "status": "待确认"})


def test_batch_keeps_same_person_in_each_source_with_independent_reviews():
    matches = VideoMatchList()
    matches.update({**event("a", 0.9, 1), "source": "photo.jpg"})
    matches.update({**event("a", 0.8, 2), "source": "video.mp4"})
    assert len(matches.ranked()) == 2
    assert len(matches) == 1
    matches.reject(1, "a")
    assert matches.ranked()[0].source == "video.mp4"
    assert matches.displayed()[-1].source == "photo.jpg"


def test_manual_association_can_be_rejected_even_without_an_automatic_candidate():
    matches = VideoMatchList()
    matches.update(event("a", 0.9))
    manual = VideoMatch("12:01:00", 1, "b", "Bob", "ID-b", 0.0, 0.8, has_score=False)
    matches.associate(manual)
    matches.reject(1, "b")
    assert matches.association(1) is None
    rejected = next(match for match in matches.displayed() if match.person_id == "b")
    assert rejected.status == "不是此人" and not rejected.has_score
    matches.associate(replace(rejected, status="人工确认"))
    assert matches.association(1).status == "人工确认"
    assert not matches.is_rejected(1, "b")
    matches.remove_association(1)
    assert [match.person_id for match in matches.displayed()] == ["a"]


def test_reviewed_sample_score_survives_later_automatic_updates():
    matches = VideoMatchList()
    matches.update(event("a", 0.7))
    matches.associate(replace(matches.ranked()[0], status="人工确认"))
    matches.update(event("a", 0.9))
    matches.reject(1, "a")
    assert matches.displayed()[0].score == 0.7
    matches.undo_rejection(1, "a")
    assert matches.displayed()[0].score == 0.9
