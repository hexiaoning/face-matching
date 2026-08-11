from face_matching.results import TargetCandidateList, VideoMatchList, format_video_time


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


def test_target_candidates_rank_strictly_by_score_and_keep_every_confidence_band():
    candidates = TargetCandidateList()
    candidates.update(
        {"time": "00:00:02.000", "track_id": 1, "decision": "review", "score": 0.31,
         "best_score": 0.34, "quality": 0.7, "support": 1}
    )
    candidates.update(
        {"time": "00:00:03.000", "track_id": 2, "decision": "confirmed", "score": 0.27,
         "best_score": 0.29, "quality": 0.6, "support": 2}
    )
    candidates.update(
        {"time": "00:00:01.000", "track_id": 3, "decision": "low", "score": 0.11,
         "best_score": 0.13, "quality": 0.5, "support": 0}
    )
    assert candidates.update(
        {"time": "00:00:04.000", "track_id": 1, "decision": "review", "score": 0.30,
         "best_score": 0.33, "quality": 0.8, "support": 1}
    ) is False
    ranked = candidates.ranked()
    assert [item.track_id for item in ranked] == [1, 2, 3]
    assert candidates.confirmed_count == 1
    assert candidates.review_count == 1
    assert candidates.low_count == 1
    assert [item.track_id for item in candidates.ranked(2)] == [1, 2]


def test_target_candidate_finalization_updates_time_range_without_replacing_best_score():
    candidates = TargetCandidateList()
    candidates.update(
        {"best_time": "00:00:02.000", "start_time": "00:00:01.000",
         "end_time": "00:00:02.000", "track_id": 7, "decision": "review",
         "score": 0.31, "best_score": 0.34, "quality": 0.7, "support": 1,
         "observations": 4, "evidence": 2}
    )

    assert candidates.update(
        {"best_time": "00:00:03.000", "start_time": "00:00:01.000",
         "end_time": "00:00:06.000", "track_id": 7, "decision": "review",
         "score": 0.29, "best_score": 0.32, "quality": 0.8, "support": 1,
         "observations": 8, "evidence": 4, "finalized": True}
    ) is True

    candidate = candidates.ranked()[0]
    assert candidate.score == 0.31
    assert candidate.best_time == "00:00:02.000"
    assert candidate.end_time == "00:00:06.000"
    assert candidate.observations == 8


def test_format_video_time_uses_media_position_and_frame_fallback():
    assert format_video_time(3_723_456) == "01:02:03.456"
    assert format_video_time(0, frame_index=75, fps=25) == "00:00:03.000"
