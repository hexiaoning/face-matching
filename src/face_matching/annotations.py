from __future__ import annotations


def annotation_states(records: list[dict]) -> dict[int, dict]:
    """Resolve each operation against the complete chronological history."""
    states = {}
    latest = {}
    for record in sorted(records, key=lambda item: item["id"]):
        action = record["action"]
        key = (record["session_id"], record["track_id"],
               "rejection" if action in {"reject", "undo_rejection"} else "association")
        if key[-1] == "rejection":
            key += (record["person_id"],)
        previous = latest.get(key)
        if previous is not None:
            states[previous]["is_current"] = False
            states[previous]["superseded_by"] = record["id"]
        states[record["id"]] = {
            "is_current": action in {"reject", "associate"}, "superseded_by": None,
        }
        latest[key] = record["id"]
    return states
