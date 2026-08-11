"""SCRFD 人脸检测器（ONNX Runtime）。

输出：bbox(x1,y1,x2,y2)、检测分数、5 点关键点。
参考 InsightFace scrfd.py 的后处理（anchor-free FCOS 风格）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config

_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2


@dataclass
class FaceDet:
    box: np.ndarray       # (4,) x1,y1,x2,y2
    score: float
    kps: np.ndarray       # (5,2) 左眼、右眼、鼻尖、左嘴角、右嘴角

    @property
    def size(self) -> float:
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        return float(min(w, h))


class SCRFDDetector:
    def __init__(self, model_path: str, providers: list[str] | None = None,
                 input_size: int = config.DET_INPUT_SIZE):
        import onnxruntime as ort

        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_size = input_size
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self._fmc = len(_STRIDES)
        self._use_kps = len(self.output_names) == self._fmc * 3
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}

    # ---- 前处理 ----
    def _preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2_resize(img, (nw, nh))
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.float32)
        canvas[:nh, :nw] = resized.astype(np.float32)
        canvas = (canvas - 127.5) / 127.5
        blob = canvas.transpose(2, 0, 1)[None]
        return blob, scale

    def _distance2bbox(self, points, distance):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points, distance):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def _centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key in self._center_cache:
            return self._center_cache[key]
        ys, xs = np.mgrid[:height, :width]
        pts = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32)
        pts = (pts * stride + (stride - 1) / 2.0)
        pts = np.repeat(pts, _NUM_ANCHORS, axis=0)
        self._center_cache[key] = pts
        return pts

    # ---- 推理 ----
    def detect(self, img: np.ndarray, score_thresh: float = 0.5,
               nms_thresh: float = 0.4, max_faces: int = 64) -> list[FaceDet]:
        blob, scale = self._preprocess(img)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(_STRIDES):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self._fmc] * stride
            height = self.input_size // stride
            width = self.input_size // stride
            pos_inds = np.where(scores >= score_thresh)[0]
            if pos_inds.size == 0:
                continue
            anchor_centers = self._centers(height, width, stride)
            pos_centers = anchor_centers[pos_inds]
            bboxes = self._distance2bbox(pos_centers, bbox_preds[pos_inds])
            pos_scores = scores[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(bboxes)
            if self._use_kps:
                kps_preds = net_outs[idx + self._fmc * 2] * stride
                kpss = self._distance2kps(pos_centers, kps_preds[pos_inds])
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss)

        if not scores_list:
            return []

        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list) / scale
        kpss = np.vstack(kpss_list) / scale if self._use_kps else None

        order = scores.argsort()[::-1]
        keep = _nms(bboxes[order], scores[order], nms_thresh)[:max_faces]
        idxs = order[keep]

        dets: list[FaceDet] = []
        for i in idxs:
            kps = kpss[i] if kpss is not None else np.zeros((5, 2), dtype=np.float32)
            dets.append(FaceDet(box=bboxes[i].astype(np.float32),
                                score=float(scores[i]),
                                kps=kps.astype(np.float32)))
        return dets


def cv2_resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> np.ndarray:
    """标准 NMS，boxes/scores 已按分数降序。"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    order = np.arange(len(scores))
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return np.array(keep, dtype=np.int64)
