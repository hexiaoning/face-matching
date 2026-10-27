from __future__ import annotations

import math
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .person_vehicle import ObjectDetection, annotate_objects


class VideoControl:
    def __init__(self) -> None:
        self.stopped = threading.Event()
        self.paused = threading.Event()

    def wait_for_resume(self) -> bool:
        while self.paused.is_set() and not self.stopped.is_set():
            self.stopped.wait(0.05)
        return not self.stopped.is_set()


@dataclass(frozen=True)
class DetectionVideoReport:
    frames: int
    fps: float
    cancelled: bool
    output: Path | None


FrameCallback = Callable[[np.ndarray, list[ObjectDetection], int, int, float], None]


def analyze_video(
    source: Path, detector: object, output: Path | None = None,
    on_frame: FrameCallback | None = None, control: VideoControl | None = None,
    realtime: bool = False,
) -> DetectionVideoReport:
    """Detect every decoded frame. Preview throttling never skips inference/export.

    MP4 exports preserve frame rate and frame order, with no audio track. Failed
    or cancelled work removes temporary video; existing destinations are safe.
    """
    control = control or VideoControl()
    if output is not None and output.exists():
        raise FileExistsError("输出文件已存在，请换一个新文件名；不会覆盖原视频或已有结果。")
    capture = cv2.VideoCapture(str(source))
    writer = None
    temporary: Path | None = None
    frames, last_preview = 0, 0
    last_emit = -math.inf
    try:
        if not capture.isOpened():
            raise ValueError("无法打开视频，请检查路径和视频编码")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("视频帧率无效，无法确定播放时间")
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        total = max(0, int(count)) if math.isfinite(count) else 0
        annotated, detections = None, []
        while control.wait_for_resume():
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                if frames == 0:
                    raise ValueError("视频没有可解码的画面")
                if total > 0 and frames < total:
                    raise ValueError(f"视频在第 {frames} 帧提前结束（预期 {total} 帧），未生成完整结果")
                break
            next_detections = detector.detect(frame)
            if control.stopped.is_set():
                break
            detections = next_detections
            annotated = annotate_objects(frame, detections)
            if output is not None:
                if writer is None:
                    with tempfile.NamedTemporaryFile(prefix=".detection-", suffix=".mp4",
                                                     dir=output.parent, delete=False) as stream:
                        temporary = Path(stream.name)
                    height, width = frame.shape[:2]
                    video_size = (width + width % 2, height + height % 2)
                    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, video_size)
                    if not writer.isOpened():
                        raise OSError("无法创建 MP4 视频编码器")
                # MPEG-4 requires even dimensions. Pad rather than drop edge pixels.
                height, width = annotated.shape[:2]
                encoded = cv2.copyMakeBorder(annotated, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                writer.write(encoded)
            frames += 1
            now = time.monotonic()
            if on_frame and now - last_emit >= 1 / 15:
                on_frame(annotated, detections, frames, total, fps)
                last_emit, last_preview = now, frames
            if realtime:
                control.stopped.wait(max(0, 1 / fps - (time.monotonic() - started)))
        cancelled = control.stopped.is_set()
        if on_frame and frames != last_preview and annotated is not None:
            on_frame(annotated, detections, frames, total, fps)
        if writer is not None:
            writer.release()
            writer = None
        if output is not None and not cancelled:
            if temporary is None:
                raise ValueError("未生成可导出的视频")
            check = cv2.VideoCapture(str(temporary))
            try:
                if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != frames:
                    raise OSError("导出视频校验失败，可能磁盘空间不足")
                check.set(cv2.CAP_PROP_POS_FRAMES, frames - 1)
                if not check.read()[0]:
                    raise OSError("导出视频末帧无法解码")
            finally:
                check.release()
            # Exclusive creation also protects a destination created during analysis.
            with output.open("xb") as destination:
                try:
                    with temporary.open("rb") as source_stream:
                        shutil.copyfileobj(source_stream, destination)
                except BaseException:
                    destination.close()
                    output.unlink(missing_ok=True)
                    raise
        return DetectionVideoReport(frames, fps, cancelled, output if not cancelled else None)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
