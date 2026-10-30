from pathlib import Path
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtTest import QTest

from face_matching.config import AppConfig
from face_matching.gui.semantic_search import SemanticSearchWidget, SourceVideoDialog, time_label
from face_matching.image_enhancement import encode_png
from face_matching.semantic_model import ChineseTokenizer, normalized_feature
from face_matching.semantic_search import COLORS, QueryScorer, SemanticIndex, object_colors, parse_query, sample_frames
from face_matching.visual_search import SearchHit, read_hit_frame


def feature(index):
    result = np.zeros(512, np.float32)
    result[index] = 1.
    return result


class Encoder:
    def __init__(self, *args):
        self.images = 0
        self.tokenizer = SimpleNamespace(encode=lambda text: {})

    def text(self, text):
        if text == '戴着帽子的人':
            return feature(1)
        if text == '没有戴帽子的人':
            return feature(0)
        color = next((color for color in COLORS if color in text), None)
        return feature(COLORS.index(color) + 1 if color else 0)

    def image(self, image):
        self.images += 1
        return feature(2)  # white, used by the deterministic index tests

    def providers(self):
        return {'text': ['test'], 'vision': ['test']}


def detector():
    return SimpleNamespace(detect=lambda frame: [SimpleNamespace(
        bbox=(0, 0, frame.shape[1], frame.shape[0]), is_person=False)])


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (96, 64))
    assert writer.isOpened()
    try:
        for index in range(6):
            writer.write(np.full((64, 96, 3), 210 + index * 5, np.uint8))
    finally:
        writer.release()


def test_chinese_combinations_and_category_are_preserved():
    query = parse_query('黑色帽子，蓝色上衣，白色裤子的人')
    assert query.category == 'person'
    assert [(c.color, c.part) for c in query.clauses] == [('黑色', 1), ('蓝色', 2), ('白色', 3)]
    assert parse_query('白色的车').category == 'vehicle'
    assert parse_query('白车').clauses[0].color == '白色'
    for text in ('', '不穿白色裤子的人', '黑色裤子白色裤子的人', '蓝衣的人和白色的车'):
        with pytest.raises(ValueError):
            parse_query(text)


def test_all_conditions_must_belong_to_same_person():
    query = parse_query('黑色帽子、蓝色上衣、白色裤子的人')
    scorer = QueryScorer(query, Encoder())
    matching = np.stack([feature(1), feature(1), feature(4), feature(2)])
    assert scorer.score(matching) == pytest.approx(1.)
    person_a = matching.copy()
    person_a[3] = feature(5)  # red trousers
    person_b = matching.copy()
    person_b[1] = feature(5)  # red hat
    assert scorer.score(person_a) is None
    assert scorer.score(person_b) is None


def test_region_color_check_rejects_semantically_similar_wrong_clothing():
    crop = np.zeros((200, 80, 3), np.uint8)
    crop[50:110] = (255, 0, 0)
    crop[116:170] = (255, 255, 255)
    colors = object_colors(crop, True)
    features = np.stack([feature(1)] * 4)
    correct = QueryScorer(parse_query('黑色帽子蓝色上衣白色裤子的人'), Encoder())
    assert correct.score(features, colors) is not None
    wrong = QueryScorer(parse_query('红色上衣白色裤子的人'), Encoder())
    assert wrong.score(features, colors) is None
    wrong = QueryScorer(parse_query('蓝色裤子的人'), Encoder())
    assert wrong.score(features, colors) is None


def test_tokenizer_wordpieces_mask_punctuation_and_no_silent_truncation(tmp_path):
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '黑', '色', '帽', '子', 'blue', '##ish', ',', 'cafe']
    path = tmp_path / 'vocab.txt'
    path.write_text('\n'.join(vocab), encoding='utf-8')
    tokenizer = ChineseTokenizer(path)
    tokens = tokenizer.encode('黑色帽子 BLUEish, café')
    assert tokens['input_ids'][0, :11].tolist() == [2, 4, 5, 6, 7, 8, 9, 10, 11, 3, 0]
    assert tokens['attention_mask'].sum() == 10
    assert not tokens['token_type_ids'].any()
    assert tokenizer.encode('unknown')['input_ids'][0, 1] == 1
    with pytest.raises(ValueError, match='截断'):
        tokenizer.encode('黑' * 51)
    with pytest.raises(ValueError):
        normalized_feature(np.full(512, np.nan))


def test_video_index_reuse_timestamps_changed_file_and_sampling(tmp_path):
    video = tmp_path / '原视频.avi'
    make_video(video)
    encoder = Encoder()
    index = SemanticIndex(tmp_path / 'index.db')
    try:
        hits, failures, stats = index.search([video], parse_query('白色的车'), encoder, detector, step=2)
        assert not failures and len(hits) == 3
        assert {h.frame_index for h in hits} == {0, 2, 4}
        assert sorted(h.seconds for h in hits) == pytest.approx([0., .2, .4])
        assert read_hit_frame(hits[1]).shape == (64, 96, 3)
        assert stats['indexed_sources'] == 1 and encoder.images == 3
        def unexpected():
            raise AssertionError('Index reuse must not initialize detection or vision')
        again, failures, stats = index.search([video], parse_query('白色汽车'), encoder, unexpected, step=2)
        assert not failures and len(again) == 3
        assert stats['reused_sources'] == 1 and encoder.images == 3
        index.search([video], parse_query('白色的车'), encoder, detector, step=3)
        assert encoder.images == 5
        video.touch()
        _, failures, stats = index.search([video], parse_query('白色的车'), encoder, detector, step=2)
        assert not failures and stats['indexed_sources'] == 1
        assert encoder.images == 8
    finally:
        index.close()


def test_cancel_rolls_back_incomplete_source_and_errors_are_reported(tmp_path):
    video = tmp_path / 'sample.avi'
    make_video(video)
    stopped = threading.Event()
    class CancellingEncoder(Encoder):
        def image(self, image):
            stopped.set()
            return super().image(image)
    index = SemanticIndex(tmp_path / 'index.db')
    try:
        hits, failures, _ = index.search([video], parse_query('白色的车'), CancellingEncoder(),
                                       detector, stopped=stopped, step=1)
        assert not hits and not failures
        assert index.connection.execute('SELECT COUNT(*) FROM sources').fetchone()[0] == 0
        assert index.connection.execute('SELECT COUNT(*) FROM objects').fetchone()[0] == 0
        stopped.clear()
        bad = tmp_path / 'bad.png'
        bad.write_bytes(b'broken')
        hits, failures, _ = index.search([bad, video], parse_query('白色的车'), Encoder(), detector, step=2, max_results=2)
        assert len(hits) == 2 and len(failures) == 1 and str(bad) in failures[0]
    finally:
        index.close()


def test_semantic_widget_worker_export_metadata_and_video_seek(qt_runtime, tmp_path, monkeypatch):
    monkeypatch.setenv('FACE_MATCHING_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr('face_matching.gui.semantic_search.SemanticEncoder', Encoder)
    monkeypatch.setattr('face_matching.gui.semantic_search.load_person_vehicle_detector', lambda *a, **kw: detector())
    video = tmp_path / 'white.avi'
    make_video(video)
    tool = SemanticSearchWidget(AppConfig())
    try:
        tool.set_paths([video])
        tool.query.setText('白色的车')
        tool.step.setValue(2)
        assert tool.start_button.isEnabled()
        tool.start()
        assert not tool.query.isEnabled()
        deadline = time.monotonic() + 10
        while tool.is_busy() and time.monotonic() < deadline:
            qt_runtime.processEvents()
            QTest.qWait(5)
        assert not tool.is_busy() and len(tool.hits) == 3
        assert tool.play_button.isEnabled()
        assert tool.search_info['description'] == '白色的车'
        assert tool.search_info['statistics']['indexed_sources'] == 1
        hit = next(hit for hit in tool.hits if hit.frame_index == 2)
        player = SourceVideoDialog(hit, tool)
        try:
            assert player.slider.value() == 2
            assert '00:00:00.200' in player.position.text()
            player.seek(4)
            assert player.slider.value() == 4
            player.toggle()
            assert player.timer.isActive()
            player.pause()
            assert not player.timer.isActive()
        finally:
            player.done(0)
            assert not player.capture.isOpened()
            player.deleteLater()
    finally:
        tool.stop()
        if tool.worker:
            tool.worker.wait(10000)
        tool.close()
        tool.deleteLater()


def test_timestamp_carry_does_not_display_sixty_seconds():
    assert time_label(59.9999) == '00:01:00.000'
    assert time_label(3661.2) == '01:01:01.200'


def test_text_encoder_feeds_only_exported_inputs_and_caches_features(monkeypatch):
    from face_matching import semantic_model

    calls = []
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name=name) for name in ('input_ids', 'attention_mask')]

        def run(self, outputs, inputs):
            assert set(inputs) == {'input_ids', 'attention_mask'}
            calls.append(inputs)
            return [feature(2)[None] * 2]

    def create(path, device, backend, **kwargs):
        assert set(kwargs['sample_inputs']) == {'input_ids', 'attention_mask'}
        return Session()

    monkeypatch.setattr(semantic_model, 'required_semantic_models', lambda root: ('vision', 'text', 'vocab'))
    monkeypatch.setattr(semantic_model, 'ChineseTokenizer', lambda path: SimpleNamespace(
        encode=lambda text: {name: np.ones((1, 52), np.int64)
                             for name in ('input_ids', 'attention_mask', 'token_type_ids')}))
    monkeypatch.setattr(semantic_model, 'create_gpu_session', create)
    encoder = semantic_model.SemanticEncoder('openvino')
    np.testing.assert_array_equal(encoder.text('白色的车'), feature(2))
    np.testing.assert_array_equal(encoder.text('白色的车'), feature(2))
    assert len(calls) == 1
