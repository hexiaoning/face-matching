from __future__ import annotations

import json
import math
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2

from .database import FaceDatabase
from .annotations import annotation_states


@dataclass(frozen=True, slots=True)
class ExportOptions:
    include_media: bool = True
    full_video: bool = False
    seconds_before: int = 5
    seconds_after: int = 5
    include_portraits: bool = True
    include_person_info: bool = True


@dataclass(frozen=True, slots=True)
class ExportReport:
    annotation_count: int
    issues: tuple[str, ...]


def write_video_clip(
    source: Path, destination: Path, frame_index: int, before: int, after: int
) -> dict:
    """Export a local, silent MP4 with offsets relative to the original video."""
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not capture.isOpened():
            raise OSError("无法读取视频")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("视频帧率无效，无法定位标注片段")
        anchor = max(0, frame_index - 1)
        start = max(0, anchor - int(before * fps))
        end = anchor + int(after * fps) + 1
        if start and not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise OSError("无法定位视频片段")
        # Some decoders seek to an earlier keyframe; decode up to the requested start.
        position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if position > start:
            raise OSError("视频定位超出标注位置")
        for _ in range(start - position):
            if not capture.grab():
                raise OSError("标注位置超出视频长度")
        count = 0
        for _ in range(start, end):
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise OSError("无法创建 MP4 视频片段")
            writer.write(frame)
            count += 1
        if count <= anchor - start:
            raise OSError("标注位置超出视频长度")
        return {"start_frame": start + 1, "frame_count": count, "fps": fps,
                "annotation_frame": anchor - start + 1, "audio_included": False}
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def export_annotation_library(
    database: FaceDatabase,
    destination: Path,
    options: ExportOptions,
    progress: Callable[[str], None] | None = None,
    *,
    annotation_ids: set[int] | None = None,
) -> ExportReport:
    if options.seconds_before < 0 or options.seconds_after < 0:
        raise ValueError("片段时长不能为负数")
    if options.include_media and not options.full_video and not (
        options.seconds_before or options.seconds_after
    ):
        raise ValueError("请设置片段前后时长")
    annotations = database.list_annotations()
    states = annotation_states(annotations)
    if annotation_ids is not None:
        available = {item["id"] for item in annotations}
        if annotation_ids - available:
            raise ValueError("选中的标注已变化，请刷新后重新选择")
        annotations = [item for item in annotations if item["id"] in annotation_ids]
    if not annotations:
        raise ValueError("没有可导出的标注")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    with tempfile.TemporaryDirectory(prefix="face-annotations-", dir=destination.parent) as folder:
        root = Path(folder)
        archive = root / "annotations.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as output:
            included: dict[tuple, tuple[str, dict]] = {}
            records = []
            for item in annotations:
                record = {key: value for key, value in item.items()
                          if key not in {"snapshot_path", "face_path"}}
                record["bbox"] = json.loads(record["bbox"]) if record["bbox"] else None
                record.update(states[item["id"]])
                record["media"] = None
                if progress:
                    progress(f"导出标注 {len(records) + 1}/{len(annotations)}")
                if options.include_media:
                    for column, label in (("snapshot_path", "snapshot"), ("face_path", "face")):
                        path = Path(item[column]) if item[column] else None
                        if path:
                            if path.is_file():
                                name = f"annotations/{item['id']}-{label}{path.suffix.lower()}"
                                output.write(path, name)
                                record[label] = name
                            else:
                                issues.append(f"标注 {item['id']}：标注画面文件缺失")
                    source = Path(item["source"])
                    if source.is_file():
                        image = source.suffix.lower() in image_extensions
                        key = (str(source.resolve()), None if image or options.full_video
                               else item["frame_index"])
                        if key not in included:
                            try:
                                if image or options.full_video:
                                    name = f"media/{item['id']}{source.suffix.lower()}"
                                    output.write(source, name)
                                    metadata = {"mode": "image" if image else "full_video"}
                                else:
                                    name = f"media/{item['id']}-clip.mp4"
                                    clip = root / f"{item['id']}-clip.mp4"
                                    metadata = write_video_clip(
                                        source, clip, item["frame_index"],
                                        options.seconds_before, options.seconds_after,
                                    )
                                    metadata["mode"] = "clip"
                                    output.write(clip, name)
                                    clip.unlink()
                                included[key] = (name, metadata)
                            except (OSError, ValueError, cv2.error) as exc:
                                issues.append(f"标注 {item['id']}：媒体导出失败（{exc}）")
                        if key in included:
                            name, metadata = included[key]
                            record["media"] = {"path": name, **metadata}
                    else:
                        issues.append(f"标注 {item['id']}：来源媒体不可用，已导出可用标注画面")
                records.append(record)

            people = []
            person_ids = sorted({item["person_id"] for item in annotations if item["person_id"]})
            samples = database.list_stored_samples() if options.include_portraits else []
            for index, person_id in enumerate(person_ids, 1):
                person = database.get_person(person_id)
                entry = {"person_id": person_id}
                if person and options.include_person_info:
                    entry.update(asdict(person))
                if options.include_portraits:
                    entry["portraits"] = []
                    for photo_index, sample in enumerate(
                        (sample for sample in samples if sample.person_id == person_id), 1
                    ):
                        path = Path(sample.image_path)
                        if path.is_file():
                            name = f"people/{index}/portraits/{photo_index}{path.suffix.lower()}"
                            output.write(path, name)
                            entry["portraits"].append(name)
                        else:
                            issues.append(f"人员 {person_id}：录入人像文件缺失")
                if options.include_person_info or options.include_portraits:
                    people.append(entry)
            output.writestr("annotations.json", json.dumps(records, ensure_ascii=False, indent=2))
            # Resolve before filtering: selecting an old label must never resurrect it.
            current = [record for record in records if record["is_current"]]
            output.writestr("current_annotations.json", json.dumps(current, ensure_ascii=False, indent=2))
            if people:
                output.writestr("people.json", json.dumps(people, ensure_ascii=False, indent=2))
            output.writestr("manifest.json", json.dumps({
                "format_version": 2, "annotation_count": len(records),
                "active_annotation_count": len(current),
                "scope": "selected" if annotation_ids is not None else "all",
                "selected_annotation_ids": sorted(annotation_ids) if annotation_ids is not None else None,
                "options": asdict(options), "issues": issues,
                "note": "片段无音轨；frame_index 为原视频中从 1 开始的帧号。有效状态依据完整历史计算；current_annotations.json 仅含导出范围内仍有效的标注。superseded_by 可引用未选中、未导出的后续操作。",
            }, ensure_ascii=False, indent=2))
        os.replace(archive, destination)
    return ExportReport(len(annotations), tuple(issues))
