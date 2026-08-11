from __future__ import annotations

import numpy as np

from face_match.config import AppSettings
from face_match.domain import FaceObservation, FaceQuality, MatchResult, TrackView
from face_match.vision.alignment import align_face
from face_match.vision.detector import ScrfdDetector
from face_match.vision.embedder import LvFaceEmbedder
from face_match.vision.matcher import MultiTemplateMatcher
from face_match.vision.quality import assess_face_quality
from face_match.vision.tracker import FaceTracker


class VideoFacePipeline:
    def __init__(
        self,
        detector: ScrfdDetector,
        embedder: LvFaceEmbedder,
        matcher: MultiTemplateMatcher,
        settings: AppSettings,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.settings = settings
        self.tracker = FaceTracker(maximum_samples=settings.maximum_track_samples)

    def reset(self) -> None:
        self.tracker.reset()

    def process(self, frame_bgr: np.ndarray) -> list[TrackView]:
        detections = self.detector.detect(
            frame_bgr, score_threshold=self.settings.detection_threshold
        )
        observations: list[FaceObservation] = []
        eligible_faces: list[np.ndarray] = []
        eligible_indices: list[int] = []
        for detection in detections:
            try:
                aligned = align_face(frame_bgr, detection.landmarks)
                quality = assess_face_quality(aligned, detection, frame_bgr.shape)
            except (ValueError, np.linalg.LinAlgError):
                aligned = None
                quality = FaceQuality(0.0, 0.0, 0.0, 0.0, 0.0)
            observation = FaceObservation(detection, quality, None, aligned)
            observations.append(observation)
            if aligned is not None and quality.overall >= self.settings.minimum_quality:
                eligible_indices.append(len(observations) - 1)
                eligible_faces.append(aligned)
        if eligible_faces:
            embeddings = self.embedder.embed(eligible_faces)
            for observation_index, embedding in zip(eligible_indices, embeddings):
                observations[observation_index].embedding = embedding

        tracks = self.tracker.update(observations)
        views: list[TrackView] = []
        for track in tracks:
            if (
                track.aggregate is None
                or len(track.embeddings) < self.settings.minimum_track_samples
            ):
                match = MatchResult.unknown(
                    f"采集中 {len(track.embeddings)}/{self.settings.minimum_track_samples}"
                )
            else:
                match = self.matcher.match(
                    track.aggregate,
                    threshold=self.settings.similarity_threshold,
                    ambiguity_margin=self.settings.ambiguity_margin,
                )
            if match.accepted and match.person_id is not None:
                if track.candidate_person_id == match.person_id:
                    track.candidate_hits += 1
                else:
                    track.candidate_person_id = match.person_id
                    track.candidate_hits = 1
                if track.candidate_hits < 2:
                    match = MatchResult.unknown("候选身份确认中", match.score, match.second_score)
            else:
                track.candidate_person_id = None
                track.candidate_hits = 0
            is_new = bool(
                match.accepted
                and match.person_id is not None
                and track.last_reported_person_id != match.person_id
            )
            if is_new:
                track.last_reported_person_id = match.person_id
            views.append(
                TrackView(
                    track_id=track.track_id,
                    bbox=track.bbox.copy(),
                    quality=track.quality,
                    sample_count=len(track.embeddings),
                    match=match,
                    is_new_event=is_new,
                )
            )
        return views


def annotate_frame(frame_bgr: np.ndarray, views: list[TrackView]) -> np.ndarray:
    import cv2

    output = frame_bgr.copy()
    for view in views:
        left, top, right, bottom = [round(float(value)) for value in view.bbox]
        if view.match.accepted:
            color = (62, 201, 115)
            label = f"#{view.track_id} MATCH {view.match.score:.3f}"
        elif view.sample_count == 0:
            color = (50, 180, 255)
            label = f"#{view.track_id} LOW QUALITY"
        else:
            color = (70, 110, 235)
            label = f"#{view.track_id} UNKNOWN {view.match.score:.3f}"
        cv2.rectangle(output, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
        text_y = max(18, top - 8)
        cv2.putText(
            output,
            label,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return output
