from face_matching.results import VideoMatchList


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
