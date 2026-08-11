from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .database import FaceDatabase
from .errors import EnrollmentError
from .gpu import create_gpu_session, resolve_gpu_backend
from .matching import GalleryMatcher, TargetMatcher
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


def reference_alignment_variants(aligned_face: np.ndarray) -> list[np.ndarray]:
    """Return conservative alignment alternatives for a one-off target photo."""
    variants = [np.ascontiguousarray(aligned_face)]
    for horizontal_shift in (-2.0, 2.0):
        matrix = np.asarray([[1.0, 0.0, horizontal_shift], [0.0, 1.0, 0.0]], dtype=np.float32)
        shifted = cv2.warpAffine(
            aligned_face,
            matrix,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        variants.append(shifted)
    variants.append(cv2.resize(aligned_face[2:110, 2:110], (112, 112), interpolation=cv2.INTER_LINEAR))
    return variants


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
        self.target_matcher: TargetMatcher | None = None
        self.target_name = "目标人物"

    def reset_video(self) -> None:
        self.tracker.reset()

    def refresh_gallery(self) -> None:
        self.matcher.refresh()
        self.tracker.invalidate_identities()

    def clear_target(self) -> None:
        self.target_matcher = None
        self.target_name = "目标人物"
        self.tracker.invalidate_identities()

    def set_target_image(self, image: np.ndarray, name: str = "目标人物") -> EnrollmentFeature:
        aligned, quality, detection = self._prepare_enrollment_face(image)
        references = self.embedder.embed_many(
            reference_alignment_variants(aligned),
            mirror_augmentation=self.mirror_augmentation,
        )
        self.target_matcher = TargetMatcher(
            references,
            threshold=self.config.target_match_threshold,
            review_threshold=self.config.target_review_threshold,
            min_support=self.config.target_min_support,
        )
        self.target_name = name.strip() or "目标人物"
        self.tracker.invalidate_identities()
        return EnrollmentFeature(references[0], quality, detection)

    def _prepare_enrollment_face(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, Quality, Detection]:
        if image is None or image.size == 0:
            raise EnrollmentError("无法读取照片")
        working = image
        offset_x = 0
        offset_y = 0
        detections = self.detector.detect(working)
        if not detections:
            # Portraits cropped tightly around the chin/forehead can fall
            # outside detector anchor priors. Add neutral photographic context
            # for detection and alignment; no synthetic facial detail is made.
            height, width = image.shape[:2]
            offset_x = max(16, int(round(width * 0.35)))
            offset_y = max(16, int(round(height * 0.35)))
            working = cv2.copyMakeBorder(
                image,
                offset_y,
                offset_y,
                offset_x,
                offset_x,
                cv2.BORDER_CONSTANT,
                value=(127, 127, 127),
            )
            detections = self.detector.detect(working)
        if not detections:
            raise EnrollmentError("照片中没有检测到人脸；请保留头部四周空间后重试")
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
        aligned = align_face(working, detection.landmarks)
        quality = assess_quality(aligned, detection)
        if quality.total < self.config.enrollment_min_quality:
            raise EnrollmentError(
                f"照片质量过低（{quality.total:.2f}），请换用更清晰或更正面的照片"
            )
        bbox = detection.bbox - np.asarray(
            [offset_x, offset_y, offset_x, offset_y], dtype=np.float32
        )
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, image.shape[1] - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, image.shape[0] - 1)
        landmarks = detection.landmarks - np.asarray([offset_x, offset_y], dtype=np.float32)
        original_detection = Detection(bbox, detection.score, landmarks)
        return aligned, quality, original_detection

    def enrollment_feature(self, image: np.ndarray) -> EnrollmentFeature:
        aligned, quality, detection = self._prepare_enrollment_face(image)
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
            if self.target_matcher is not None:
                track.aggregate(
                    self.config.track_top_k,
                    self.config.track_consistency_threshold,
                )
                track.apply_target_match(
                    self.target_matcher.match(track.observations), self.target_name
                )
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
