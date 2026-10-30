"""Offline Chinese-CLIP encoders; no hub client or runtime downloads."""
from __future__ import annotations

from pathlib import Path
import unicodedata

import cv2
import numpy as np

from .gpu import create_gpu_session
from .models import SEMANTIC_MODEL_ID, required_semantic_models


class ChineseTokenizer:
    """BERT basic tokenization + greedy WordPiece with the pinned vocabulary."""

    def __init__(self, path: Path) -> None:
        self.vocab = {word: index for index, word in enumerate(path.read_text(encoding='utf-8').splitlines())}

    def encode(self, text: str) -> dict[str, np.ndarray]:
        cleaned = []
        for char in text:
            code = ord(char)
            if char.isspace():
                cleaned.append(' ')
            elif code in (0, 0xfffd) or unicodedata.category(char).startswith('C'):
                continue
            elif (0x3400 <= code <= 0x4dbf or 0x4e00 <= code <= 0x9fff
                  or 0x20000 <= code <= 0x2fa1f or 0xf900 <= code <= 0xfaff
                  or unicodedata.category(char).startswith('P')
                  or 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126):
                cleaned.extend((' ', char, ' '))
            else:
                cleaned.append(char)
        basic = ''.join(char for char in unicodedata.normalize('NFD', ''.join(cleaned).lower())
                        if unicodedata.category(char) != 'Mn').split()
        pieces = []
        for token in basic:
            if len(token) > 100:
                pieces.append('[UNK]')
                continue
            start, words = 0, []
            while start < len(token):
                end = len(token)
                while end > start:
                    part = ('##' if start else '') + token[start:end]
                    if part in self.vocab:
                        break
                    end -= 1
                if end == start:
                    words = ['[UNK]']
                    break
                words.append(part)
                start = end
            pieces.extend(words)
        if len(pieces) > 50:
            raise ValueError('描述过长，请缩短到50个词元以内，避免条件被截断。')
        ids = [self.vocab['[CLS]'], *[self.vocab.get(p, self.vocab['[UNK]']) for p in pieces], self.vocab['[SEP]']]
        mask = [1] * len(ids) + [0] * (52 - len(ids))
        ids += [self.vocab['[PAD]']] * (52 - len(ids))
        return {'input_ids': np.asarray([ids], np.int64),
                'attention_mask': np.asarray([mask], np.int64),
                'token_type_ids': np.zeros((1, 52), np.int64)}


def normalized_feature(output: np.ndarray) -> np.ndarray:
    value = np.asarray(output, np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if value.size != 512 or not np.isfinite(value).all() or norm < 1e-8:
        raise ValueError('语义模型输出无效，需要512维有限特征。')
    return value / norm


class SemanticEncoder:
    model_id = SEMANTIC_MODEL_ID

    def __init__(self, backend: str = 'auto', device_id: int = 0, root: Path | None = None) -> None:
        self.vision_path, self.text_path, vocab = required_semantic_models(root)
        self.backend, self.device_id = backend, device_id
        self.tokenizer = ChineseTokenizer(vocab)
        self._text = self._vision = None
        self._texts: dict[str, np.ndarray] = {}

    def text(self, text: str) -> np.ndarray:
        if text not in self._texts:
            tokens = {name: value for name, value in self.tokenizer.encode(text).items()
                      if name in ('input_ids', 'attention_mask')}
            if self._text is None:
                self._text = create_gpu_session(self.text_path, self.device_id, self.backend,
                                               sample_inputs=tokens)
            feed = {item.name: tokens[item.name] for item in self._text.get_inputs()}
            self._texts[text] = normalized_feature(self._text.run(None, feed)[0])
        return self._texts[text]

    def image(self, image: np.ndarray) -> np.ndarray:
        if self._vision is None:
            self._vision = create_gpu_session(self.vision_path, self.device_id, self.backend,
                                             input_shape=(1, 3, 224, 224))
        rgb = cv2.cvtColor(cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        value = rgb.astype(np.float32) / 255
        value = (value - np.asarray([.48145466, .4578275, .40821073], np.float32)) / np.asarray(
            [.26862954, .26130258, .27577711], np.float32)
        tensor = np.ascontiguousarray(value.transpose(2, 0, 1)[None])
        return normalized_feature(self._vision.run(None, {self._vision.get_inputs()[0].name: tensor})[0])

    def providers(self) -> dict[str, list[str]]:
        return {key: session.get_providers() for key, session in (('text', self._text), ('vision', self._vision))
                if session is not None}
