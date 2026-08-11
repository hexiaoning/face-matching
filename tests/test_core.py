"""核心 pipeline 单元测试（CPU provider，仅验证逻辑正确性）。"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_match import config
from face_match.detector import SCRFDDetector
from face_match.recognizer import FaceRecognizer, align_face
from face_match import quality as quality_mod
from face_match.tracker import IoUTracker, Track
from face_match.database import PersonDB
from face_match.matcher import GalleryIndex
from face_match.pipeline import FacePipeline, embed_gallery_photo

ASSETS = Path(__file__).parent / "assets"
MODELS = Path(__file__).parent.parent / "models"
CPU = ["CPUExecutionProvider"]

SAME_A = ASSETS / "Rob_Lowe_0001.jpg"
SAME_B = ASSETS / "Rob_Lowe_0002.jpg"
OTHER = ASSETS / "Nathalie_Baye_0002.jpg"

pytestmark = pytest.mark.skipif(
    not (MODELS / config.DETECTOR_MODEL).is_file(), reason="模型未下载")


@pytest.fixture(scope="module")
def detector():
    return SCRFDDetector(str(MODELS / config.DETECTOR_MODEL), providers=CPU)


@pytest.fixture(scope="module")
def recognizer():
    return FaceRecognizer(str(MODELS / config.RECOGNIZER_MODEL), providers=CPU)


def _embed(detector, recognizer, path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    assert img is not None, f"读图失败 {path}"
    emb, q, _ = embed_gallery_photo(detector, recognizer, img)
    return emb


def test_detect_faces(detector):
    img = cv2.imread(str(SAME_A))
    dets = detector.detect(img)
    assert len(dets) >= 1
    det = max(dets, key=lambda d: d.size)
    assert det.score > 0.5
    assert det.kps.shape == (5, 2)
    assert det.size > config.MIN_FACE_SIZE


def test_same_person_similarity(detector, recognizer):
    ea = _embed(detector, recognizer, SAME_A)
    eb = _embed(detector, recognizer, SAME_B)
    ec = _embed(detector, recognizer, OTHER)
    same = float(ea @ eb)
    diff = float(ea @ ec)
    assert same > diff, f"同人相似度({same:.3f})应高于异人({diff:.3f})"
    assert same > 0.35, f"同人相似度过低: {same:.3f}"


def test_quality_degrades_with_blur(detector, recognizer):
    img = cv2.imread(str(SAME_A))
    dets = detector.detect(img)
    det = max(dets, key=lambda d: d.size)
    aligned = align_face(img, det.kps)
    q_clear = quality_mod.quality_score(det, aligned)

    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    dets_b = detector.detect(blurred, score_thresh=0.3)
    assert dets_b, "模糊图应仍能检测到脸"
    det_b = max(dets_b, key=lambda d: d.size)
    aligned_b = align_face(blurred, det_b.kps)
    q_blur = quality_mod.quality_score(det_b, aligned_b)
    assert q_blur < q_clear, f"模糊图质量({q_blur:.3f})应低于清晰图({q_clear:.3f})"


def test_blurred_face_still_matches(detector, recognizer):
    """监控场景核心：模糊帧的 embedding 融合后仍应匹配到本人。"""
    ea = _embed(detector, recognizer, SAME_A)
    img = cv2.imread(str(SAME_B))
    # 模拟监控：缩小 + 模糊 + 噪声
    small = cv2.resize(img, None, fx=0.5, fy=0.5)
    noisy = cv2.GaussianBlur(small, (5, 5), 0)
    noisy = np.clip(noisy.astype(np.float32) + np.random.normal(0, 8, noisy.shape), 0, 255).astype(np.uint8)
    dets = detector.detect(noisy, score_thresh=0.3)
    assert dets
    det = max(dets, key=lambda d: d.size)
    aligned = align_face(noisy, det.kps)
    eb = recognizer.embed_aligned(aligned)
    sim = float(ea @ eb)
    assert sim > 0.25, f"模糊退化后相似度过低: {sim:.3f}"


def test_tracker_fusion():
    tr = IoUTracker()
    emb_good = np.random.RandomState(0).rand(512).astype(np.float32)
    emb_good /= np.linalg.norm(emb_good)
    box = np.array([100, 100, 200, 200], dtype=np.float32)
    # 同一位置 5 帧，质量递增
    for i in range(5):
        noise = np.random.RandomState(i + 1).rand(512).astype(np.float32) * (0.3 - 0.05 * i)
        e = emb_good + noise
        e /= np.linalg.norm(e)
        tr.update(i, [(box.copy(), e, 0.3 + 0.1 * i)])
    tracks = tr.active()
    assert len(tracks) == 1
    fused = tracks[0].fused_embedding()
    assert fused is not None
    # 融合结果应比最差单帧更接近真实 embedding
    assert float(fused @ emb_good) > 0.9


def test_db_and_matcher(tmp_path, detector, recognizer):
    os.environ["FACEMATCH_DATA_DIR"] = str(tmp_path)
    import importlib
    importlib.reload(config)
    try:
        db = PersonDB(tmp_path / "t.db")
        pid1 = db.add_person("张三", "110101199001011234")
        pid2 = db.add_person("李四", "110101199002022345")
        pa = db.add_photo(pid1, str(SAME_A))
        pb = db.add_photo(pid1, str(SAME_B))
        pc = db.add_photo(pid2, str(OTHER))
        for ph_id, path in [(pa, SAME_A), (pb, SAME_B), (pc, OTHER)]:
            db.set_photo_embedding(ph_id, _embed(detector, recognizer, path), 0.9)

        persons = db.list_persons()
        assert len(persons) == 2
        assert len(persons[0].photos) == 2

        idx = GalleryIndex()
        idx.rebuild(db.gallery())
        assert len(idx) == 3
        probe = _embed(detector, recognizer, SAME_B)
        m = idx.best_match(probe, threshold=0.35)
        assert m is not None and m.person.name == "张三", f"匹配错误: {m}"

        # 把与 probe 同源的照片移出库后再测：跨照片匹配与阈值过滤
        db.delete_photo(pb)
        idx.rebuild(db.gallery())
        m = idx.best_match(probe, threshold=0.35)
        assert m is not None and m.person.name == "张三", \
            f"同人的另一张照片应仍能匹配: {m}"
        # 阈值高于跨照片相似度时不应误报
        assert idx.best_match(probe, threshold=m.score + 0.01) is None
        db.close()
    finally:
        importlib.reload(config)


def test_pipeline_frame(detector, recognizer):
    idx = GalleryIndex()
    db_emb = _embed(detector, recognizer, SAME_A)
    from face_match.database import Person, Photo
    p = Person(id=1, name="张三", id_number="", created_at="",
               photos=[Photo(id=1, path=str(SAME_A), embedding=db_emb, quality=0.9)])
    idx.rebuild([(p, db_emb)])

    pipe = FacePipeline(detector, recognizer, idx, threshold=0.30)
    img = cv2.imread(str(SAME_B))
    result = None
    # 同一张图送多帧，让 track 积累足够样本
    for i in range(3):
        result = pipe.process_frame(img, i, i / 25.0)
    assert result.tracks, "应检测到人脸 track"
    trk = result.tracks[0]
    assert trk.fused_ready
    assert trk.match is not None and trk.match.person.name == "张三", \
        f"pipeline 应识别出张三, got {trk.match}"
