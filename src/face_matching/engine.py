from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AppConfig
from .database import FaceDatabase
from .errors import EnrollmentError
from .gpu import create_gpu_session, resolve_gpu_backend
from .matching import GalleryMatcher, TargetMatcher
from .models import feature_model_id, profile_spec, required_paths
from .vision.alignment import align_face, align_face_variants
from .vision.detector import Detection, SCRFDDetector
from .vision.quality import Quality, assess_quality
from .vision.recognizer import FaceEmbedder
from .vision.tracker import FaceTracker, Observation, bbox_iou


@dataclass(frozen=True, slots=True)
class EnrollmentFeature:
    embedding: np.ndarray
    quality: Quality
    detection: Detection


@dataclass(frozen=True, slots=True)
class SearchTarget:
    name: str
    embeddings: tuple[np.ndarray, ...]
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
    decision: str = "low"
    support: int = 0
    best_score: float = 0.0
    evidence: int = 0
    last_frame: int = 0


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
        self.search_target: SearchTarget | None = None
        self.target_matcher: TargetMatcher | None = None

    def reset_video(self) -> None:
        self.tracker.reset()

    def refresh_gallery(self) -> None:
        self.matcher.refresh()
        self.tracker.invalidate_identities()

    def set_search_target(self, image: np.ndarray, name: str = "目标人物") -> SearchTarget:
        detection = self._primary_enrollment_detection(image)
        aligned = align_face(image, detection.landmarks)
        quality = assess_quality(aligned, detection)
        if quality.total < self.config.enrollment_min_quality:
            raise EnrollmentError(
                f"目标照片质量过低（{quality.total:.2f}），请换用更清晰或更正面的照片"
            )
        # A tight portrait can yield a different five-point estimate with and
        # without neutral context.  Keep both overlapping hypotheses as
        # reference templates instead of trusting the higher detector score
        # blindly; this is a one-time cost and does not add confirmation votes.
        template_detections = [detection]
        direct = self.detector.detect(
            image,
            threshold=min(self.config.detector_threshold, 0.25),
        )
        for candidate in direct:
            if bbox_iou(candidate.bbox, detection.bbox) < 0.45:
                continue
            landmark_delta = float(
                np.mean(
                    np.linalg.norm(
                        np.asarray(candidate.landmarks) - np.asarray(detection.landmarks),
                        axis=1,
                    )
                )
            )
            if landmark_delta >= 0.5:
                template_detections.append(candidate)
                break
        variants = [
            aligned
            for item in template_detections
            for aligned in align_face_variants(image, item.landmarks)
        ]
        embeddings = tuple(
            self.embedder.embed_many(
                variants,
                mirror_augmentation=self.mirror_augmentation,
            )
        )
        target = SearchTarget(name.strip() or "目标人物", embeddings, quality, detection)
        self.search_target = target
        self.target_matcher = TargetMatcher(
            list(embeddings),
            threshold=self.config.target_match_threshold,
            review_threshold=self.config.target_review_threshold,
            min_support=self.config.target_min_support,
            min_evidence_gap=self.config.target_min_evidence_gap,
            consistency_threshold=self.config.track_consistency_threshold,
            auto_confirm=self.config.target_auto_confirm,
        )
        self.tracker.reset()
        return target

    def clear_search_target(self) -> None:
        self.search_target = None
        self.target_matcher = None
        self.tracker.reset()

    def _primary_enrollment_detection(self, image: np.ndarray) -> Detection:
        # A single supplied target may be a stylized or tightly cropped photo.
        # Use a recall-oriented enrollment cutoff; video detection keeps the
        # configured threshold to control the much larger frame volume.
        detections = self.detector.detect_reference(image)
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
        return detection

    def enrollment_feature(self, image: np.ndarray) -> EnrollmentFeature:
        detection = self._primary_enrollment_detection(image)
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

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float | None = None,
    ) -> FrameResult:
        detections = self.detector.detect(frame)
        media_timestamp = float(frame_index) if timestamp is None else float(timestamp)
        observations: list[Observation] = []
        aligned_faces: list[np.ndarray] = []
        embedding_groups: list[tuple[int, int, int]] = []
        for detection in detections:
            quality_score = 0.0
            pose_score = 0.0
            if min(detection.width, detection.height) >= self.config.min_face_size:
                try:
                    aligned = align_face(frame, detection.landmarks)
                    quality = assess_quality(aligned, detection)
                    quality_score = quality.total
                    pose_score = quality.pose
                    if quality.total >= self.config.min_quality:
                        if self.target_matcher is not None and (
                            quality.pose < 0.75 or detection.score < 0.65
                        ):
                            variants = align_face_variants(
                                frame,
                                detection.landmarks,
                                horizontal_offsets=(-2.0, 0.0),
                            )
                            start = len(aligned_faces)
                            aligned_faces.extend(variants)
                            # The official alignment (offset 0) remains the
                            # primary tracking embedding; offset -2 is target
                            # search evidence for uncertain landmarks.
                            embedding_groups.append((len(observations), start + 1, 2))
                        else:
                            start = len(aligned_faces)
                            aligned_faces.append(aligned)
                            embedding_groups.append((len(observations), start, 1))
                except (ValueError, FloatingPointError):
                    pass
            observations.append(
                Observation(
                    bbox=detection.bbox,
                    detection_score=detection.score,
                    embedding=None,
                    quality=quality_score,
                    timestamp=media_timestamp,
                    pose=pose_score,
                )
            )
        embeddings = self.embedder.embed_many(
            aligned_faces,
            mirror_augmentation=self.mirror_augmentation,
        )
        for observation_index, primary_index, group_size in embedding_groups:
            observations[observation_index].embedding = embeddings[primary_index]
            group_start = primary_index - (group_size - 1)
            observations[observation_index].alternate_embeddings = tuple(
                embeddings[index]
                for index in range(group_start, group_start + group_size)
                if index != primary_index
            )
        tracks = self.tracker.update(observations, frame_index)
        for track in tracks:
            if self.target_matcher is not None and self.search_target is not None:
                if track.embedding_version != track.matched_embedding_version:
                    track.apply_target_match(
                        self.target_matcher.match(track.target_observations),
                        self.search_target.name,
                    )
                continue
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
                decision=track.decision,
                support=track.support,
                best_score=track.best_score,
                evidence=track.evidence,
                last_frame=track.last_frame,
            )
            for track in tracks
        )
        return FrameResult(views, len(detections), len(embedding_groups))
