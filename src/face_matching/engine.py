from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AppConfig
from .database import FaceDatabase
from .errors import EnrollmentError
from .gpu import create_gpu_session, resolve_gpu_backend
from .matching import GalleryMatcher
from .models import feature_model_id, profile_spec, required_paths
from .vision.alignment import align_face
from .vision.detector import Detection, SCRFDDetector
from .vision.quality import Quality, assess_quality
from .vision.recognizer import FaceEmbedder
from .vision.tracker import FaceTracker, Observation


@dataclass(frozen=True, slots=True)
class EnrollmentFeature:
    embedding: np.ndarray
    quality: Quality
    detection: Detection


@dataclass(frozen=True, slots=True)
class TrackView:
    id: int
    bbox: tuple[float, float, float, float]
    name: str
    person_id: str | None
    id_card: str
    score: float
    quality: float
    observations: int


@dataclass(frozen=True, slots=True)
class FrameResult:
    tracks: tuple[TrackView, ...]
    detected_faces: int
    usable_faces: int


class FaceEngine:
    """GPU inference plus quality-aware temporal template matching."""

    def __init__(
        self,
        config: AppConfig,
        database: FaceDatabase,
        model_root: Path | None = None,
    ) -> None:
        self.config = config.validate()
        self.database = database
        self.profile = profile_spec(config.model_profile)
        self.mirror_augmentation = bool(config.mirror_augmentation)
        self.model_id = feature_model_id(config.model_profile, self.mirror_augmentation)
        self.gpu_backend = resolve_gpu_backend(config.gpu_backend, config.gpu_device_id)
        detector_path, recognizer_path = required_paths(config.model_profile, model_root)
        detector_session = create_gpu_session(
            detector_path, config.gpu_device_id, backend=self.gpu_backend
        )
        recognizer_session = create_gpu_session(
            recognizer_path, config.gpu_device_id, backend=self.gpu_backend
        )
        self.detector = SCRFDDetector(
            detector_session,
            input_size=config.detector_size,
            threshold=config.detector_threshold,
            nms_threshold=config.nms_threshold,
        )
        self.embedder = FaceEmbedder(recognizer_session)
        self.matcher = GalleryMatcher(
            database,
            self.model_id,
            threshold=config.match_threshold,
            min_margin=config.match_margin,
        )
        self.tracker = FaceTracker(max_misses=config.track_max_misses)

    def reset_video(self) -> None:
        self.tracker.reset()

    def refresh_gallery(self) -> None:
        self.matcher.refresh()
        self.tracker.invalidate_identities()

    def enrollment_feature(self, image: np.ndarray) -> EnrollmentFeature:
        detections = self.detector.detect(image)
        if not detections:
            raise EnrollmentError("照片中没有检测到人脸")
        ranked = sorted(
            detections,
            key=lambda item: item.width * item.height * item.score,
            reverse=True,
        )
        detection = ranked[0]
        primary_area = max(detection.width * detection.height, 1.0)
        if any(
            min(item.width, item.height) >= self.config.min_face_size
            and item.width * item.height >= primary_area * 0.25
            for item in ranked[1:]
        ):
            raise EnrollmentError("照片中有多张明显人脸，请使用只包含本人的照片")
        if min(detection.width, detection.height) < self.config.min_face_size:
            raise EnrollmentError(
                f"人脸过小（{min(detection.width, detection.height):.0f}px），请使用更清晰的照片"
            )
        aligned = align_face(image, detection.landmarks)
        quality = assess_quality(aligned, detection)
        if quality.total < self.config.enrollment_min_quality:
            raise EnrollmentError(
                f"照片质量过低（{quality.total:.2f}），请换用更清晰或更正面的照片"
            )
        return EnrollmentFeature(
            self.embedder.embed(aligned, mirror_augmentation=self.mirror_augmentation),
            quality,
            detection,
        )

    def process_frame(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        detections = self.detector.detect(frame)
        observations: list[Observation] = []
        aligned_faces: list[np.ndarray] = []
        usable_observation_indexes: list[int] = []
        for detection in detections:
            quality_score = 0.0
            if min(detection.width, detection.height) >= self.config.min_face_size:
                try:
                    aligned = align_face(frame, detection.landmarks)
                    quality = assess_quality(aligned, detection)
                    quality_score = quality.total
                    if quality.total >= self.config.min_quality:
                        usable_observation_indexes.append(len(observations))
                        aligned_faces.append(aligned)
                except (ValueError, FloatingPointError):
                    pass
            observations.append(
                Observation(
                    bbox=detection.bbox,
                    detection_score=detection.score,
                    embedding=None,
                    quality=quality_score,
                )
            )
        embeddings = self.embedder.embed_many(
            aligned_faces,
            mirror_augmentation=self.mirror_augmentation,
        )
        for observation_index, embedding in zip(usable_observation_indexes, embeddings, strict=True):
            observations[observation_index].embedding = embedding
        tracks = self.tracker.update(observations, frame_index)
        for track in tracks:
            if len(track.observations) < self.config.min_track_observations:
                continue
            if track.embedding_version == track.matched_embedding_version:
                continue
            aggregate = track.aggregate(
                self.config.track_top_k,
                self.config.track_consistency_threshold,
            )
            if aggregate is not None:
                track.apply_match(
                    self.matcher.match(aggregate),
                    consensus=self.config.confirmation_matches,
                )
        views = tuple(
            TrackView(
                id=track.id,
                bbox=tuple(float(value) for value in track.bbox),
                name=track.name,
                person_id=track.person_id,
                id_card=track.id_card,
                score=track.score,
                quality=track.quality,
                observations=len(track.observations),
            )
            for track in tracks
        )
        return FrameResult(views, len(detections), len(embeddings))
