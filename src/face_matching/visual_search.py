"""Offline query-by-example search over local images and video frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import threading

import cv2
import numpy as np

from .vision.io import IMAGE_EXTENSIONS, read_image

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.m4v', '.wmv', '.webm'}


@dataclass(frozen=True)
class SearchHit:
    source: str
    frame_index: int
    seconds: float | None
    score: float
    bbox: tuple[int, int, int, int]
    method: str


def collect_sources(paths: list[Path]) -> list[Path]:
    sources: dict[str, Path] = {}
    for path in paths:
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        for item in candidates:
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                sources.setdefault(str(item.resolve()).casefold(), item.resolve())
    return list(sources.values())


class ObjectExampleMatcher:
    """Find the same visible instance using geometry and multiscale appearance.

    This is not semantic category search or person identity recognition.
    """
    def __init__(self, query: np.ndarray, threshold: float = 0.75) -> None:
        if min(query.shape[:2]) < 12:
            raise ValueError("目标选区过小，宽和高至少为12像素")
        self.threshold = threshold
        self.query = self._resize(query, 480)[0]
        self.gray = cv2.cvtColor(self.query, cv2.COLOR_BGR2GRAY)
        if self.gray.std() < 3:
            raise ValueError("目标选区缺少可区分纹理，请扩大选区或更换图片")
        self.sift = cv2.SIFT_create(nfeatures=2000)
        self.points, self.descriptors = self.sift.detectAndCompute(self.gray, None)

    @staticmethod
    def _resize(image: np.ndarray, maximum: int) -> tuple[np.ndarray, float]:
        scale = min(1., maximum / max(image.shape[:2]))
        if scale < 1:
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale
        return image, 1.

    def find(self, frame: np.ndarray) -> list[tuple[float, tuple[int, int, int, int], str]]:
        image, scale = self._resize(frame, 1600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        candidates = []
        if self.descriptors is not None and len(self.points) >= 6:
            points, descriptors = self.sift.detectAndCompute(gray, None)
            if descriptors is not None and len(descriptors) >= 2:
                pairs = cv2.BFMatcher().knnMatch(self.descriptors, descriptors, k=2)
                good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
                if len(good) >= 6:
                    origin = np.float32([self.points[m.queryIdx].pt for m in good])
                    target = np.float32([points[m.trainIdx].pt for m in good])
                    matrix, mask = cv2.findHomography(origin, target, cv2.RANSAC, 3.)
                    if matrix is not None and mask is not None and np.isfinite(matrix).all():
                        qh, qw = self.gray.shape
                        corners = np.float32([[0, 0], [qw, 0], [qw, qh], [0, qh]]).reshape(-1, 1, 2)
                        projected = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
                        count = int(mask.sum())
                        hull = cv2.convexHull(origin[mask.ravel().astype(bool)])
                        coverage = cv2.contourArea(hull) / max(qw * qh, 1)
                        area = abs(cv2.contourArea(projected))
                        if (count >= 6 and coverage >= .04 and np.isfinite(projected).all()
                                and cv2.isContourConvex(projected) and 144 <= area <= width * height * 1.1):
                            x1, y1 = np.maximum(projected.min(axis=0), 0)
                            x2, y2 = np.minimum(projected.max(axis=0), (width, height))
                            score = min(.99, .5 + .25 * count / len(good) + .25 * min(count / 20, 1))
                            if x2 > x1 and y2 > y1:
                                candidates.append((score, (x1, y1, x2, y2), '局部特征与几何校验'))
        # Also find texture-poor or exact crops; avoid undefined correlation
        # for flat templates and suppress image-border false positives.
        qh, qw = self.gray.shape
        for factor in (.35, .5, .7, 1., 1.4, 2., 2.8):
            tw, th = round(qw * factor), round(qh * factor)
            if min(tw, th) < 12 or tw > width or th > height:
                continue
            template = cv2.resize(self.gray, (tw, th))
            if template.std() < 3:
                continue
            scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            candidates.append((float(score), (x, y, x + tw, y + th), '多尺度外观匹配'))
        if not candidates:
            return []
        score, box, method = max(candidates, key=lambda candidate: candidate[0])
        if score < self.threshold:
            return []
        bbox = tuple(int(round(value / scale)) for value in box)
        return [(score, bbox, method)]


class FaceExampleMatcher:
    def __init__(self, query: np.ndarray, engine: object, threshold: float) -> None:
        self.engine, self.threshold = engine, threshold
        self.embedding = engine.enrollment_feature(query).embedding

    def find(self, frame: np.ndarray) -> list[tuple[float, tuple[int, int, int, int], str]]:
        from .vision.alignment import align_face
        from .vision.quality import assess_quality

        usable = []
        for detection in self.engine.detector.detect(frame):
            if min(detection.width, detection.height) < self.engine.config.min_face_size:
                continue
            aligned = align_face(frame, detection.landmarks)
            if assess_quality(aligned, detection).total >= self.engine.config.min_quality:
                usable.append((detection, aligned))
        embeddings = self.engine.embedder.embed_many([image for _, image in usable],
                                                     mirror_augmentation=self.engine.mirror_augmentation)
        result = []
        for (detection, _), embedding in zip(usable, embeddings):
            score = float(np.dot(self.embedding, embedding))
            if score >= self.threshold:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = detection.bbox
                bbox = (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
                result.append((score, bbox, '人脸特征相似候选'))
        return result


def search_sources(sources: list[Path], matcher: object, *, frame_step: int = 1,
                   stopped: threading.Event | None = None, max_results: int = 500,
                   on_progress: Callable[[str], None] | None = None) -> tuple[list[SearchHit], list[str]]:
    if frame_step < 1 or max_results < 1:
        raise ValueError("抽帧间隔和结果上限必须大于0")
    stopped = stopped or threading.Event()
    hits: list[SearchHit] = []
    failures = []

    def inspect(frame: np.ndarray, source: Path, index: int, seconds: float | None) -> None:
        for score, bbox, method in matcher.find(frame):
            hits.append(SearchHit(str(source), index, seconds, score, bbox, method))
        if len(hits) > max_results:
            hits.sort(key=lambda hit: hit.score, reverse=True)
            del hits[max_results:]

    for number, source in enumerate(sources, 1):
        if stopped.is_set():
            break
        try:
            if on_progress:
                on_progress(f"搜索 {number}/{len(sources)}：{source.name}")
            if source.suffix.lower() in IMAGE_EXTENSIONS:
                image = read_image(source)
                if image is None:
                    raise ValueError("无法解码图片")
                inspect(image, source, 0, None)
                continue
            capture = cv2.VideoCapture(str(source))
            try:
                if not capture.isOpened():
                    raise ValueError("无法打开视频")
                fps = capture.get(cv2.CAP_PROP_FPS)
                index, decoded = 0, 0
                while not stopped.is_set():
                    if not capture.grab():
                        break
                    if index % frame_step == 0:
                        ok, frame = capture.retrieve()
                        if not ok:
                            raise ValueError(f"无法解码第 {index + 1} 帧")
                        decoded += 1
                        inspect(frame, source, index, index / fps if np.isfinite(fps) and fps > 0 else None)
                        if on_progress and decoded % 10 == 0:
                            on_progress(f"搜索 {number}/{len(sources)}：{source.name} · 第 {index + 1} 帧")
                    index += 1
                if not decoded and not stopped.is_set():
                    raise ValueError("没有可解码的视频帧")
            finally:
                capture.release()
        except Exception as exc:
            failures.append(f"{source}: {exc}")
    return sorted(hits, key=lambda hit: hit.score, reverse=True), failures


def read_hit_frame(hit: SearchHit) -> np.ndarray:
    if Path(hit.source).suffix.lower() in IMAGE_EXTENSIONS:
        frame = read_image(hit.source)
    else:
        capture = cv2.VideoCapture(hit.source)
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, hit.frame_index)
            ok, frame = capture.read()
            if not ok:
                frame = None
        finally:
            capture.release()
    if frame is None:
        raise ValueError("无法读取搜索结果的来源画面，文件可能已被移动")
    return frame
