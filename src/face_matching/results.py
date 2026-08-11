from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


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
