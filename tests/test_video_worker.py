from face_matching.gui.main_window import VideoWorker, _final_preview_due, _rate_limit_due


def test_rate_limit_allows_first_value_and_enforces_interval():
    assert _rate_limit_due(None, now=10.0, interval=0.1)
    assert not _rate_limit_due(10.0, now=10.09, interval=0.1)
    assert _rate_limit_due(10.0, now=10.11, interval=0.1)


def test_video_preview_is_limited_to_fifteen_fps():
    assert VideoWorker.PREVIEW_INTERVAL_SECONDS == 1.0 / 15.0


def test_final_preview_is_emitted_only_after_normal_end_when_latest_frame_was_skipped():
    assert _final_preview_due(reached_end=True, frame_index=100, last_preview_frame_index=90)
    assert not _final_preview_due(reached_end=True, frame_index=100, last_preview_frame_index=100)
    assert not _final_preview_due(reached_end=False, frame_index=100, last_preview_frame_index=90)
    assert not _final_preview_due(reached_end=True, frame_index=0, last_preview_frame_index=0)
