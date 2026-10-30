"""Persistent per-object semantic index, Chinese clothing/vehicle queries."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
from typing import Callable

import cv2
import numpy as np

from .models import PERSON_VEHICLE_MODEL, SEMANTIC_MODEL_ID
from .paths import data_dir
from .visual_search import SearchHit
from .vision.io import IMAGE_EXTENSIONS, read_image


COLORS = ('黑色', '白色', '灰色', '蓝色', '红色', '绿色', '黄色', '棕色', '紫色', '粉色', '橙色')
PARTS = {'帽子': 1, '帽': 1, '上衣': 2, '衣服': 2, '外套': 2, '夹克': 2,
         '衬衫': 2, '短袖': 2, 'T恤': 2, '卫衣': 2, '裤子': 3, '裤': 3, '长裤': 3, '短裤': 3, '裙子': 3}
INDEX_VERSION = 'objects-v3-det035-resize224-hsv'


@dataclass(frozen=True)
class ColorClause:
    color: str
    noun: str
    part: int


@dataclass(frozen=True)
class SemanticQuery:
    text: str
    category: str
    clauses: tuple[ColorClause, ...]


def parse_query(text: str, category: str = 'auto') -> SemanticQuery:
    text = text.strip()
    if not text or len(text) > 160:
        raise ValueError('请输入1～160字的人物衣着或车辆颜色描述。')
    if category not in ('auto', 'person', 'vehicle'):
        raise ValueError('未知搜索目标类型')
    if re.search(r'不穿|不是|没有|不戴|非白|非黑|除了|或者|或是', text):
        raise ValueError('当前请使用肯定描述，每次搜索一种衣着组合或车辆颜色。')
    has_vehicle = bool(re.search(r'车|汽车|轿车|货车|公交|摩托', text))
    has_person = bool(re.search(r'人|帽|衣|裤|裙|外套|夹克|衬衫|短袖|卫衣|T恤', text, re.I))
    if category == 'auto':
        if has_vehicle and has_person:
            raise ValueError('请将人物衣着和车辆分开搜索，或明确选择搜索目标。')
        category = 'vehicle' if has_vehicle else 'person'
    elif (category == 'person' and has_vehicle and not has_person) or (category == 'vehicle' and has_person and not has_vehicle):
        raise ValueError('文字描述与选择的目标类型不同，请修改描述或选择自动判断。')
    noun_pattern = '|'.join(sorted(PARTS, key=len, reverse=True)) if category == 'person' else r'车辆|汽车|轿车|货车|公交车|摩托车|自行车|车'
    color_pattern = '|'.join(c[:-1] for c in COLORS)
    matches = re.finditer(f'({color_pattern})(?:色)?(?:的)?\\s*({noun_pattern})', text, re.I)
    clauses = tuple(ColorClause(match[1] + '色', match[2], PARTS.get(match[2], 0)) for match in matches)
    by_part = {}
    for clause in clauses:
        if clause.part in by_part and by_part[clause.part] != clause.color:
            raise ValueError('同一区域出现多种颜色，请拆成多次搜索。')
        by_part[clause.part] = clause.color
    return SemanticQuery(text, category, clauses)


class QueryScorer:
    def __init__(self, query: SemanticQuery, encoder: object) -> None:
        self.query = query
        self.whole = encoder.text(query.text)
        self.attributes = []
        self.hat_presence = None
        for clause in query.clauses:
            alternatives = np.stack([encoder.text(color + clause.noun) for color in COLORS])
            self.attributes.append((clause, COLORS.index(clause.color), alternatives))
            if clause.part == 1:
                self.hat_presence = np.stack([encoder.text('戴着帽子的人'), encoder.text('没有戴帽子的人')])

    def score(self, features: np.ndarray, colors: np.ndarray | None = None) -> float | None:
        whole = float(features[0] @ self.whole)
        if self.hat_presence is not None:
            presence = self.hat_presence @ features[1]
            if float(presence[0]) < float(presence[1]) + .005:
                return None
        scores = []
        for clause, target, alternatives in self.attributes:
            similarities = alternatives @ features[clause.part]
            # Each clause is evaluated on a region of the SAME detected object.
            # Allow a small color margin for lighting changes; this is a candidate filter.
            if colors is None:
                if float(similarities[target]) < float(similarities.max()) - .025:
                    return None
            else:
                # Require visible local color as well as semantic similarity.
                minimum = .18 if clause.color in ('黑色', '白色', '灰色') else .10
                if clause.color == '白色' and self.query.category == 'person':
                    minimum = .55  # gaps between dark trouser legs can expose a bright background
                if float(colors[clause.part, target]) < minimum:
                    return None
            scores.append(float(similarities[target]))
        return whole if not scores else .4 * whole + .4 * float(np.mean(scores)) + .2 * min(scores)


def object_features(crop: np.ndarray, person: bool, encoder: object) -> np.ndarray:
    full = encoder.image(crop)
    if not person:
        return np.stack([full] * 4)
    height = crop.shape[0]
    features = [full]
    for start, end in ((0., .30), (.18, .62), (.52, 1.)):
        region = crop[int(height * start):max(int(height * start) + 1, int(height * end))]
        features.append(encoder.image(region))
    return np.stack(features).astype(np.float32)


def object_colors(crop: np.ndarray, person: bool) -> np.ndarray:
    """Coarse visible colors in central body regions; no inferred night colors."""
    height, width = crop.shape[:2]
    regions = ((.1, .9, .1, .9), (.2, .8, .0, .23), (.2, .8, .25, .55), (.25, .75, .58, .85))
    result = []
    for left, right, top, bottom in regions:
        if not person:
            left, right, top, bottom = regions[0]
        area = crop[int(height * top):max(int(height * top) + 1, int(height * bottom)),
                    int(width * left):max(int(width * left) + 1, int(width * right))]
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue, saturation, value = hsv[..., 0], hsv[..., 1] / 255., hsv[..., 2] / 255.
        chromatic = (saturation >= .22) & (value > .18)
        masks = (
            value <= .27,
            (saturation < .22) & (value >= .72),
            (saturation < .22) & (value > .27) & (value < .72),
            chromatic & (hue >= 90) & (hue < 130),
            chromatic & ((hue < 10) | (hue >= 170)) & (value >= .35),
            chromatic & (hue >= 35) & (hue < 90),
            chromatic & (hue >= 20) & (hue < 35),
            chromatic & (hue >= 5) & (hue < 30) & (value < .65),
            chromatic & (hue >= 130) & (hue < 155),
            chromatic & (hue >= 155) & (hue < 175) & (value >= .5),
            chromatic & (hue >= 10) & (hue < 20) & (value >= .65),
        )
        result.append([float(mask.mean()) for mask in masks])
    return np.asarray(result, np.float32)


def source_signature(path: Path) -> str:
    info = path.stat()
    return f'{info.st_size}:{info.st_mtime_ns}:{getattr(info, "st_ino", 0)}'


def sample_frames(path: Path, step: int, stopped: threading.Event):
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = read_image(path)
        if frame is None:
            raise ValueError('无法解码图片')
        yield 0, None, frame
        return
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('无法打开视频')
        fps = capture.get(cv2.CAP_PROP_FPS)
        previous_time = -1.
        index, decoded = 0, 0
        while not stopped.is_set() and capture.grab():
            if index % step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError(f'无法解码第{index + 1}帧')
                milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
                seconds = milliseconds / 1000
                if not math.isfinite(seconds) or seconds < 0 or (index > 0 and seconds <= previous_time):
                    if not math.isfinite(fps) or fps <= 0:
                        raise ValueError('视频没有有效时间戳或帧率，无法可靠定位时间点')
                    seconds = index / fps
                previous_time = seconds
                decoded += 1
                yield index, seconds, frame
            index += 1
        if not decoded and not stopped.is_set():
            raise ValueError('视频没有可解码的帧')
    finally:
        capture.release()


class SemanticIndex:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or data_dir() / 'semantic-index.db'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute('PRAGMA foreign_keys=ON')
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY, path TEXT NOT NULL, signature TEXT NOT NULL,
                pipeline TEXT NOT NULL, step INTEGER NOT NULL, UNIQUE(path, pipeline, step)
            );
            CREATE TABLE IF NOT EXISTS objects (
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                frame_index INTEGER NOT NULL, seconds REAL, bbox TEXT NOT NULL,
                category TEXT NOT NULL, features BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS object_sources ON objects(source_id, category);
        ''')

    def close(self) -> None:
        self.connection.close()

    def search(self, sources: list[Path], query: SemanticQuery, encoder: object, detector_factory: Callable,
               *, step: int = 25, threshold: float = .20, stopped: threading.Event | None = None,
               progress: Callable[[str], None] | None = None, max_results: int = 500) -> tuple[list[SearchHit], list[str], dict]:
        if step < 1 or not -1 <= threshold <= 1 or max_results < 1:
            raise ValueError('抽帧间隔、相关度门槛或结果上限无效')
        stopped = stopped or threading.Event()
        report = progress or (lambda message: None)
        scorer = QueryScorer(query, encoder)
        pipeline = SEMANTIC_MODEL_ID + ':' + INDEX_VERSION + ':' + PERSON_VEHICLE_MODEL.sha256
        detector = None
        hits, failures = [], []
        stats = {'reused_sources': 0, 'indexed_sources': 0, 'sampled_frames': 0, 'objects': 0}
        for number, source in enumerate(sources, 1):
            if stopped.is_set():
                break
            source = source.resolve()
            try:
                signature = source_signature(source)
                row = self.connection.execute('SELECT id, signature FROM sources WHERE path=? AND pipeline=? AND step=?',
                                              (str(source), pipeline, step)).fetchone()
                if row is None or row[1] != signature:
                    if detector is None:
                        report('正在加载离线人车检测模型…')
                        detector = detector_factory()
                    report(f'建立索引 {number}/{len(sources)}：{source.name}')
                    with self.connection:
                        if row:
                            self.connection.execute('DELETE FROM sources WHERE id=?', (row[0],))
                        cursor = self.connection.execute('INSERT INTO sources(path,signature,pipeline,step) VALUES(?,?,?,?)',
                                                         (str(source), signature, pipeline, step))
                        source_id = cursor.lastrowid
                        for frame_index, seconds, frame in sample_frames(source, step, stopped):
                            if stopped.is_set():
                                raise InterruptedError()
                            stats['sampled_frames'] += 1
                            height, width = frame.shape[:2]
                            for detection in detector.detect(frame):
                                if stopped.is_set():
                                    raise InterruptedError()
                                x1, y1, x2, y2 = detection.bbox
                                box = (max(0, int(x1)), max(0, int(y1)), min(width, math.ceil(x2)), min(height, math.ceil(y2)))
                                if box[2] <= box[0] or box[3] <= box[1]:
                                    continue
                                crop = frame[box[1]:box[3], box[0]:box[2]]
                                features = object_features(crop, detection.is_person, encoder)
                                values = np.concatenate((features, object_colors(crop, detection.is_person)), axis=1)
                                self.connection.execute('INSERT INTO objects VALUES(?,?,?,?,?,?)',
                                    (source_id, frame_index, seconds, json.dumps(box),
                                     'person' if detection.is_person else 'vehicle', values.astype('<f4').tobytes()))
                                stats['objects'] += 1
                            report(f'建立索引 {number}/{len(sources)}：{source.name} · 第{frame_index + 1}帧')
                        if stopped.is_set():
                            raise InterruptedError()
                        if source_signature(source) != signature:
                            raise ValueError('分析期间源文件发生变化，请重新搜索')
                    stats['indexed_sources'] += 1
                else:
                    source_id = row[0]
                    stats['reused_sources'] += 1
                report(f'检索 {number}/{len(sources)}：{source.name}')
                cursor = self.connection.execute('SELECT frame_index,seconds,bbox,features FROM objects WHERE source_id=? AND category=?',
                                                 (source_id, query.category))
                for frame_index, seconds, box, payload in cursor:
                    if stopped.is_set():
                        break
                    values = np.frombuffer(payload, dtype='<f4').reshape(4, 512 + len(COLORS))
                    score = scorer.score(values[:, :512], values[:, 512:])
                    if score is not None and score >= threshold:
                        hits.append(SearchHit(str(source), frame_index, seconds, score, tuple(json.loads(box)), '中文语义候选'))
                    if len(hits) > max_results * 2:
                        hits.sort(key=lambda hit: hit.score, reverse=True)
                        del hits[max_results:]
            except InterruptedError:
                break
            except Exception as exc:
                failures.append(f'{source}: {exc}')
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:max_results], failures, stats
