from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .database import Database
from .domain import MatchResult, Person, PhotoRecord, normalize_embedding


@dataclass(slots=True)
class _PersonTemplates:
    person: Person
    photo_ids: np.ndarray
    embeddings: np.ndarray
    prototype: np.ndarray


class GalleryIndex:
    """In-memory, per-person multi-template cosine index."""

    def __init__(self) -> None:
        self._templates: list[_PersonTemplates] = []
        self.photo_count = 0

    @property
    def person_count(self) -> int:
        return len(self._templates)

    def rebuild(self, rows: list[tuple[PhotoRecord, Person]]) -> None:
        grouped: dict[int, list[tuple[PhotoRecord, Person]]] = defaultdict(list)
        for photo, person in rows:
            grouped[person.id].append((photo, person))
        templates: list[_PersonTemplates] = []
        for entries in grouped.values():
            person = entries[0][1]
            vectors = np.stack([normalize_embedding(entry[0].embedding) for entry in entries])
            qualities = np.asarray(
                [max(0.1, float(entry[0].quality)) for entry in entries], dtype=np.float32
            )
            prototype = normalize_embedding(np.average(vectors, axis=0, weights=qualities))
            templates.append(
                _PersonTemplates(
                    person=person,
                    photo_ids=np.asarray([entry[0].id for entry in entries], dtype=np.int64),
                    embeddings=np.ascontiguousarray(vectors, dtype=np.float32),
                    prototype=prototype,
                )
            )
        self._templates = templates
        self.photo_count = sum(len(template.photo_ids) for template in templates)

    def reload(self, database: Database) -> None:
        self.rebuild(database.gallery_rows())

    def match(self, embedding: np.ndarray) -> MatchResult | None:
        if not self._templates:
            return None
        query = normalize_embedding(embedding)
        best_result: MatchResult | None = None
        for templates in self._templates:
            photo_scores = templates.embeddings @ query
            best_index = int(np.argmax(photo_scores))
            best_photo_score = float(photo_scores[best_index])
            prototype_score = float(templates.prototype @ query)
            # A strong single pose and the robust multi-photo centroid both contribute.
            score = 0.70 * best_photo_score + 0.30 * prototype_score
            if best_result is None or score > best_result.score:
                best_result = MatchResult(
                    person_id=templates.person.id,
                    name=templates.person.name,
                    id_number=templates.person.id_number,
                    score=float(score),
                    best_photo_id=int(templates.photo_ids[best_index]),
                )
        return best_result

