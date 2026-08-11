"""端到端：合成监控视频（模糊+缩放+噪声+多人交替出现）→ pipeline 识别。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from face_match import config
from face_match.detector import SCRFDDetector
from face_match.recognizer import FaceRecognizer
from face_match.database import Person, Photo
from face_match.matcher import GalleryIndex
from face_match.pipeline import FacePipeline, embed_gallery_photo

ASSETS = Path(__file__).parent / "assets"
MODELS = Path(__file__).parent.parent / "models"
CPU = ["CPUExecutionProvider"]

pytestmark = pytest.mark.skipif(
    not (MODELS / config.DETECTOR_MODEL).is_file(), reason="模型未下载")


def _degrade(img: np.ndarray, rng: np.random.RandomState, scale: float = 0.6) -> np.ndarray:
    """模拟监控画质：缩小、模糊、噪声、亮度变化。"""
    out = cv2.resize(img, None, fx=scale, fy=scale)
    k = rng.choice([3, 5])
    out = cv2.GaussianBlur(out, (k, k), 0)
    noise = rng.normal(0, 6, out.shape)
    out = np.clip(out.astype(np.float32) + noise + rng.uniform(-25, 25), 0, 255)
    return out.astype(np.uint8)


def _compose_frame(rng: np.random.RandomState, faces: list[np.ndarray]) -> np.ndarray:
    """把若干人脸贴到灰色背景上，模拟监控画面。"""
    canvas = np.full((480, 720, 3), rng.randint(30, 90), dtype=np.uint8)
    for i, face in enumerate(faces):
        h, w = face.shape[:2]
        y = 100 + rng.randint(0, 200)
        x = 60 + i * 300 + rng.randint(-30, 30)
        x = min(x, 720 - w - 10)
        canvas[y:y + h, x:x + w] = face
    return canvas


def test_video_end_to_end(tmp_path):
    rng = np.random.RandomState(42)
    img_a = cv2.imread(str(ASSETS / "Rob_Lowe_0001.jpg"))   # 库照片
    img_b = cv2.imread(str(ASSETS / "Rob_Lowe_0002.jpg"))   # 视频中的本人
    img_c = cv2.imread(str(ASSETS / "Nathalie_Baye_0002.jpg"))  # 视频中的陌生人

    # 合成 60 帧：前 30 帧只有陌生人，后 30 帧本人+陌生人同框
    video_path = tmp_path / "cam.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (720, 480))
    for i in range(60):
        faces = [_degrade(img_c, rng)]
        if i >= 30:
            faces.append(_degrade(img_b, rng))
        writer.write(_compose_frame(rng, faces))
    writer.release()

    detector = SCRFDDetector(str(MODELS / config.DETECTOR_MODEL), providers=CPU)
    recognizer = FaceRecognizer(str(MODELS / config.RECOGNIZER_MODEL), providers=CPU)

    emb_a, q, _ = embed_gallery_photo(detector, recognizer, img_a)
    person = Person(id=1, name="目标人员", id_number="110101199001011234", created_at="",
                    photos=[Photo(id=1, path="", embedding=emb_a, quality=q)])
    gallery = GalleryIndex()
    gallery.rebuild([(person, emb_a)])

    pipe = FacePipeline(detector, recognizer, gallery, threshold=0.30)

    cap = cv2.VideoCapture(str(video_path))
    identified: dict[str, float] = {}
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % 2:  # 跳帧
            continue
        result = pipe.process_frame(frame, idx, idx / 10.0)
        for trk in result.tracks:
            if trk.match is not None:
                name = trk.match.person.name
                identified[name] = max(identified.get(name, 0.0), trk.match.score)
    cap.release()

    assert "目标人员" in identified, \
        f"应识别出库中人员（监控画质退化后）, identified={identified}"
    assert identified["目标人员"] > 0.30
