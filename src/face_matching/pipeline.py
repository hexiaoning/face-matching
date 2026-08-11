from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .config import EngineConfig
from .engine import FaceEngine
from .matcher import GalleryMatcher, MatchResult
from .tracker import FaceTrack, MultiFaceTracker


@dataclass(frozen=True, slots=True)
class VisualFace:
    bbox: tuple[float, float, float, float]
    track_id: int
    name: str
    masked_id_card: str
    score: float
    quality: float
    state: str
    accepted: bool


@dataclass(frozen=True, slots=True)
class RecognitionEvent:
    time: datetime
    track_id: int
    name: str
    masked_id_card: str
    score: float


@dataclass(frozen=True, slots=True)
class FrameResult:
    faces: list[VisualFace]
    events: list[RecognitionEvent]


class VideoFacePipeline:
    def __init__(
        self,
        engine: FaceEngine,
        matcher: GalleryMatcher,
        config: EngineConfig | None = None,
    ) -> None:
        self.engine = engine
        self.matcher = matcher
        self.config = config or engine.config
        self.tracker = MultiFaceTracker(
            max_age=self.config.max_track_age,
            max_embeddings=self.config.max_track_embeddings,
        )
        self.frame_index = 0
        self.match_threshold = self.config.match_threshold
        self._gallery_revision = matcher.revision

    def reset(self) -> None:
        self.tracker.reset()
        self.frame_index = 0
        self._gallery_revision = self.matcher.revision

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        self.frame_index += 1
        detections = self.engine.detect(frame_bgr)
        assignments = self.tracker.update(detections)
        if self.matcher.revision != self._gallery_revision:
            for track in self.tracker.tracks.values():
                track.invalidate_identity()
            self._gallery_revision = self.matcher.revision

        pending: list[tuple[FaceTrack, object]] = []
        crops: list[np.ndarray] = []
        for track, detection in assignments:
            quality = float(detection.quality.overall) if detection.quality else 0.0
            width = float(detection.bbox[2] - detection.bbox[0])
            height = float(detection.bbox[3] - detection.bbox[1])
            due = self.frame_index - track.last_embedding_frame >= self.config.recognition_interval
            if (
                due
                and min(width, height) >= self.config.min_face_size
                and quality >= self.config.min_quality
                and detection.aligned is not None
            ):
                pending.append((track, detection))
                crops.append(detection.aligned)
        if crops:
            embeddings = self.engine.embed(crops)
            for (track, detection), embedding in zip(pending, embeddings, strict=True):
                track.add_embedding(embedding, float(detection.quality.overall), self.frame_index)

        events: list[RecognitionEvent] = []
        for track, detection in assignments:
            if (
                len(track.embeddings) >= self.config.min_track_embeddings
                and track.embedding_version != track.matched_embedding_version
            ):
                aggregate = track.aggregate_embedding(
                    self.config.track_top_k,
                    self.config.track_similarity_floor,
                )
                if aggregate is not None:
                    match = self.matcher.match(
                        aggregate, self.match_threshold, self.config.min_margin
                    )
                    became_stable = track.update_identity(
                        match, self.config.confirmation_frames
                    )
                    if became_stable and track.stable_match is not None:
                        events.append(_event(track.id, track.stable_match))

        visual_faces: list[VisualFace] = []
        for track, detection in assignments:
            quality = float(detection.quality.overall) if detection.quality else 0.0
            name = "未知"
            masked_id = ""
            score = 0.0
            accepted = False
            state = "采集中"
            if quality < self.config.min_quality:
                state = "质量不足"
            if track.stable_match is not None:
                stable = track.stable_match
                name = stable.name
                masked_id = stable.masked_id_card
                score = stable.score
                accepted = True
                state = "已确认"
            elif track.last_match is not None:
                score = track.last_match.score
                state = "待确认" if track.last_match.accepted else "未知"
            visual_faces.append(VisualFace(
                bbox=tuple(float(value) for value in detection.bbox),
                track_id=track.id,
                name=name,
                masked_id_card=masked_id,
                score=score,
                quality=quality,
                state=state,
                accepted=accepted,
            ))
        return FrameResult(visual_faces, events)


def _event(track_id: int, match: MatchResult) -> RecognitionEvent:
    return RecognitionEvent(
        time=datetime.now(),
        track_id=track_id,
        name=match.name,
        masked_id_card=match.masked_id_card,
        score=match.score,
    )
