from __future__ import annotations

from dataclasses import dataclass, replace
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
    status: str = "自动匹配"
    reason: str = ""


class VideoMatchList:
    """Keep each person's best video match and rank the result by score."""

    def __init__(self) -> None:
        self._matches: dict[tuple[int, str], VideoMatch] = {}
        self._rejected: set[tuple[int, str]] = set()
        self._associated: dict[int, VideoMatch] = {}

    def clear(self) -> None:
        self._matches.clear()
        self._rejected.clear()
        self._associated.clear()

    def update(self, event: Mapping[str, object]) -> bool:
        match = VideoMatch(
            time=str(event["time"]),
            track_id=int(event["track_id"]),
            person_id=str(event["person_id"]),
            name=str(event["name"]),
            id_card=str(event["id_card"]),
            score=float(event["score"]),
            quality=float(event["quality"]),
            status=str(event.get("status", "自动匹配")),
            reason=str(event.get("reason", "")),
        )
        key = (match.track_id, match.person_id)
        if key in self._rejected:
            return False
        previous = self._matches.get(key)
        if previous is not None and (self._priority(previous), previous.score) >= (
            self._priority(match), match.score
        ):
            return False
        before = self.displayed()
        self._matches[key] = match
        return before != self.displayed()

    def reject(self, track_id: int, person_id: str) -> None:
        self._rejected.add((track_id, person_id))

    def undo_rejection(self, track_id: int, person_id: str) -> None:
        self._rejected.discard((track_id, person_id))

    def is_rejected(self, track_id: int, person_id: str) -> bool:
        return (track_id, person_id) in self._rejected

    def associate(self, match: VideoMatch) -> None:
        self._associated[match.track_id] = replace(match, status="人工关联")

    def remove_association(self, track_id: int) -> None:
        self._associated.pop(track_id, None)

    def association(self, track_id: int) -> VideoMatch | None:
        return self._associated.get(track_id)

    @staticmethod
    def _priority(match: VideoMatch) -> int:
        return {"待确认": 0, "自动匹配": 1, "人工关联": 2}.get(match.status, 0)

    def ranked(self) -> tuple[VideoMatch, ...]:
        best: dict[str, VideoMatch] = {}
        candidates = [
            match for key, match in self._matches.items()
            if key not in self._rejected and match.track_id not in self._associated
        ] + list(self._associated.values())
        for match in candidates:
            previous = best.get(match.person_id)
            # A manually verified identity takes precedence over an automatic score.
            if previous is None or (self._priority(match), match.score) > (
                self._priority(previous), previous.score
            ):
                best[match.person_id] = match
        return tuple(sorted(
            best.values(), key=lambda item: (-item.score, item.name, item.person_id)
        ))

    def displayed(self) -> tuple[VideoMatch, ...]:
        rejected = sorted(
            (replace(self._matches[key], status="不是此人")
             for key in self._rejected if key in self._matches),
            key=lambda item: (-item.score, item.track_id, item.person_id),
        )
        return self.ranked() + tuple(rejected)

    def __len__(self) -> int:
        return sum(match.status != "待确认" for match in self.ranked())

    @property
    def pending_count(self) -> int:
        return sum(match.status == "待确认" for match in self.ranked())
