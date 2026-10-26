"""Pluggable text embedding backends.

Both backends return L2-normalised float32 matrices so that Chroma cosine
retrieval is comparable across implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class TextEmbedder(Protocol):
    name: str
    model_name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class HashingEmbedder:
    """Deterministic local word n-gram embedding with no model download."""

    dimension: int = 2048
    name: str = "hashing"
    model_name: str = "word-ngram-hashing"

    def __post_init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=self.dimension,
            alternate_sign=False,
            norm="l2",
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(list(texts)).toarray()
        return _normalise(matrix)


class SentenceTransformerEmbedder:
    """Sentence-transformers backend loaded lazily to keep base install light."""

    name = "sentence_transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "sentence-transformers is not installed; install the 'semantic' extra"
            ) from exc
        self.model_name = model_name
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as first_error:
            # An expired Hugging Face login can make the hub lookup fail even
            # when the model is already present in the local cache.
            try:
                self._model = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                raise first_error
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"could not determine embedding dimension for {model_name}")
        self.dimension = int(dimension)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _normalise(matrix)


def create_embedder(
    backend: str,
    *,
    dimension: int = 2048,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> TextEmbedder:
    if backend == "hashing":
        return HashingEmbedder(dimension=dimension)
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(
        f"unknown embedding backend {backend!r}; choose 'hashing' or "
        "'sentence_transformers'"
    )

