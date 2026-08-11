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
    time: str
    track_id: int
    decision: str
    score: float
    best_score: float
    quality: float
    support: int


class TargetCandidateList:
    """Keep the strongest evidence image for every candidate track."""

    def __init__(self) -> None:
        self._by_track: dict[int, TargetCandidate] = {}

    def clear(self) -> None:
        self._by_track.clear()

    def update(self, event: Mapping[str, object]) -> bool:
        candidate = TargetCandidate(
            time=str(event["time"]),
            track_id=int(event["track_id"]),
            decision=str(event["decision"]),
            score=float(event["score"]),
            best_score=float(event.get("best_score", event["score"])),
            quality=float(event["quality"]),
            support=int(event["support"]),
        )
        previous = self._by_track.get(candidate.track_id)
        priority = {"confirmed": 2, "review": 1, "collecting": 0}
        if previous is not None:
            upgraded = priority.get(candidate.decision, 0) > priority.get(previous.decision, 0)
            if not upgraded and previous.score >= candidate.score:
                return False
        self._by_track[candidate.track_id] = candidate
        return True

    def ranked(self) -> tuple[TargetCandidate, ...]:
        priority = {"confirmed": 2, "review": 1, "collecting": 0}
        return tuple(
            sorted(
                self._by_track.values(),
                key=lambda item: (-priority.get(item.decision, 0), -item.score, item.track_id),
            )
        )

    @property
    def confirmed_count(self) -> int:
        return sum(item.decision == "confirmed" for item in self._by_track.values())

    @property
    def review_count(self) -> int:
        return sum(item.decision == "review" for item in self._by_track.values())

    def __len__(self) -> int:
        return len(self._by_track)
