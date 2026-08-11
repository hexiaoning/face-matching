from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def format_video_time(milliseconds: float, frame_index: int = 0, fps: float = 0.0) -> str:
    """Return a stable media timestamp, falling back to frame/fps when needed."""
    value = float(milliseconds)
    if value <= 0.0 and frame_index > 0 and fps > 0.0:
        value = frame_index * 1000.0 / fps
    value = max(value, 0.0)
    total_ms = int(round(value))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


@dataclass(frozen=True, slots=True)
class VideoMatch:
    time: str
    track_id: int
    person_id: str
    name: str
    id_card: str
    score: float
    quality: float


class VideoMatchList:
    """Keep each person's best video match and rank the result by score."""

    def __init__(self) -> None:
        self._best_by_person: dict[str, VideoMatch] = {}

    def clear(self) -> None:
        self._best_by_person.clear()

    def update(self, event: Mapping[str, object]) -> bool:
        match = VideoMatch(
            time=str(event["time"]),
            track_id=int(event["track_id"]),
            person_id=str(event["person_id"]),
            name=str(event["name"]),
            id_card=str(event["id_card"]),
            score=float(event["score"]),
            quality=float(event["quality"]),
        )
        previous = self._best_by_person.get(match.person_id)
        if previous is not None and previous.score >= match.score:
            return False
        self._best_by_person[match.person_id] = match
        return True

    def ranked(self) -> tuple[VideoMatch, ...]:
        return tuple(
            sorted(
                self._best_by_person.values(),
                key=lambda item: (-item.score, item.name, item.person_id),
            )
        )

    def __len__(self) -> int:
        return len(self._best_by_person)


@dataclass(frozen=True, slots=True)
class TargetCandidate:
    best_time: str
    start_time: str
    end_time: str
    track_id: int
    decision: str
    score: float
    best_score: float
    quality: float
    support: int
    observations: int
    evidence: int


class TargetCandidateList:
    """Keep the strongest evidence image for every candidate track."""

    def __init__(self) -> None:
        self._by_track: dict[int, TargetCandidate] = {}

    def clear(self) -> None:
        self._by_track.clear()

    def update(self, event: Mapping[str, object]) -> bool:
        candidate = TargetCandidate(
            best_time=str(event.get("best_time", event.get("time", ""))),
            start_time=str(event.get("start_time", event.get("time", ""))),
            end_time=str(event.get("end_time", event.get("time", ""))),
            track_id=int(event["track_id"]),
            decision=str(event["decision"]),
            score=float(event["score"]),
            best_score=float(event.get("best_score", event["score"])),
            quality=float(event["quality"]),
            support=int(event["support"]),
            observations=int(event.get("observations", event.get("evidence", 0))),
            evidence=int(event.get("evidence", event.get("observations", 0))),
        )
        previous = self._by_track.get(candidate.track_id)
        if previous is not None:
            priority = {"low": 0, "review": 1, "confirmed": 2}
            use_new_score = (
                candidate.score > previous.score + 1e-9
            )
            use_new_best_frame = candidate.best_score > previous.best_score + 1e-9
            upgraded = (
                priority.get(candidate.decision, 0)
                > priority.get(previous.decision, 0)
            )
            update_progress = (
                use_new_score
                or use_new_best_frame
                or upgraded
                or bool(event.get("finalized", False))
            )
            candidate = TargetCandidate(
                best_time=(
                    candidate.best_time if use_new_best_frame else previous.best_time
                ),
                start_time=previous.start_time,
                end_time=(
                    candidate.end_time
                    if update_progress
                    else previous.end_time
                ),
                track_id=candidate.track_id,
                decision=(
                    candidate.decision
                    if priority.get(candidate.decision, 0)
                    >= priority.get(previous.decision, 0)
                    else previous.decision
                ),
                score=candidate.score if use_new_score else previous.score,
                best_score=max(candidate.best_score, previous.best_score),
                quality=(
                    candidate.quality if use_new_best_frame else previous.quality
                ),
                support=(
                    max(candidate.support, previous.support)
                    if update_progress
                    else previous.support
                ),
                observations=(
                    max(candidate.observations, previous.observations)
                    if update_progress
                    else previous.observations
                ),
                evidence=(
                    max(candidate.evidence, previous.evidence)
                    if update_progress
                    else previous.evidence
                ),
            )
            if candidate == previous:
                return False
        self._by_track[candidate.track_id] = candidate
        return True

    def ranked(self, limit: int | None = None) -> tuple[TargetCandidate, ...]:
        values = sorted(
            self._by_track.values(),
            key=lambda item: (-item.score, -item.best_score, -item.quality, item.track_id),
        )
        if limit is not None:
            values = values[: max(0, int(limit))]
        return tuple(values)

    @property
    def confirmed_count(self) -> int:
        return sum(item.decision == "confirmed" for item in self._by_track.values())

    @property
    def review_count(self) -> int:
        return sum(item.decision == "review" for item in self._by_track.values())

    @property
    def low_count(self) -> int:
        return sum(item.decision == "low" for item in self._by_track.values())

    def __len__(self) -> int:
        return len(self._by_track)
