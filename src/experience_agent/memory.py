"""FAISS-backed experience memory with deterministic local hashed embeddings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only in minimal environments
    faiss = None


def _memory_text(record: dict[str, Any]) -> str:
    rule = str(record.get("rule", "")).strip()
    trigger = str(record.get("trigger", "")).strip()
    if rule and trigger:
        return f"{trigger}. {rule}"
    return rule or trigger


@dataclass
class SearchHit:
    record: dict[str, Any]
    score: float


@dataclass
class ExperienceMemory:
    dimension: int = 2048
    records: list[dict[str, Any]] = field(default_factory=list)
    _index: Any = field(default=None, init=False, repr=False)
    _vectors: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=self.dimension,
            alternate_sign=False,
            norm="l2",
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
        )
        self._index = faiss.IndexFlatIP(self.dimension) if faiss is not None else None
        self._vectors = np.empty((0, self.dimension), dtype=np.float32)

    def _embed(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts).astype(np.float32)
        embeddings = matrix.toarray()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def add(self, record: dict[str, Any]) -> bool:
        """Add a record unless an identical memory text already exists."""
        text = _memory_text(record)
        if not text:
            raise ValueError("memory record has no rule or trigger text")
        fingerprint = _normalise(text)
        if any(_normalise(_memory_text(existing)) == fingerprint for existing in self.records):
            return False
        row = self._embed([text])
        if self._index is not None:
            self._index.add(row)
        else:
            self._vectors = np.vstack([self._vectors, row])
        enriched = dict(record)
        enriched["memory_text"] = text
        self.records.append(enriched)
        return True

    def search(self, query: str, k: int = 2, min_similarity: float = 0.0) -> list[SearchHit]:
        if not self.records:
            return []
        row = self._embed([query])
        k = min(k, len(self.records))
        if self._index is not None:
            scores, indexes = self._index.search(row, k)
            pairs = zip(indexes[0].tolist(), scores[0].tolist())
        else:
            scores = (self._vectors @ row[0]).tolist()
            ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:k]
            pairs = ranked
        hits = []
        for index, score in pairs:
            if index < 0 or float(score) < min_similarity:
                continue
            hits.append(SearchHit(self.records[int(index)], float(score)))
        return hits

    def __len__(self) -> int:
        return len(self.records)

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

