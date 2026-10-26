"""基于 ChromaDB 和确定性本地哈希向量的经验记忆。

by AI.Coding
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from .embedding import create_embedder


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
    """用 Chroma 集合保存经验，并通过可替换的 embedding backend 检索。"""

    dimension: int = 2048
    embedding_backend: str = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: str | Path | None = None
    collection_name: str = "experience_memory"
    records: list[dict[str, Any]] = field(default_factory=list)
    _client: Any = field(default=None, init=False, repr=False)
    _collection: Any = field(default=None, init=False, repr=False)
    _records_by_id: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """建立 Chroma 客户端、embedding backend 和集合，并恢复已有记录。"""
        self._embedder = create_embedder(
            self.embedding_backend,
            dimension=self.dimension,
            model_name=self.embedding_model,
        )
        if self.persist_directory is None:
            self._client = chromadb.EphemeralClient()
            # 进程内的临时客户端共享状态，默认名称必须唯一以免不同维度互相冲突。
            effective_collection_name = (
                f"experience_memory_{uuid.uuid4().hex}"
                if self.collection_name == "experience_memory"
                else self.collection_name
            )
        else:
            # 持久目录由每次实验独占，避免不同运行的经验相互污染。
            directory = Path(self.persist_directory)
            directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(directory))
            effective_collection_name = self.collection_name
        self._collection = self._client.get_or_create_collection(
            name=effective_collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={
                "embedding_backend": self._embedder.name,
                "embedding_model": self._embedder.model_name,
                "embedding_dimension": self._embedder.dimension,
            },
        )
        self._verify_embedding_metadata()
        self._load_existing_records()

    def _verify_embedding_metadata(self) -> None:
        metadata = dict(self._collection.metadata or {})
        expected = {
            "embedding_backend": self._embedder.name,
            "embedding_model": self._embedder.model_name,
            "embedding_dimension": self._embedder.dimension,
        }
        for key, value in expected.items():
            if key in metadata and metadata[key] != value:
                raise ValueError(
                    f"Chroma collection {self.collection_name!r} uses {key}={metadata[key]!r}, "
                    f"but the new configuration requests {value!r}. Use a new run directory or "
                    "collection name."
                )

    def _load_existing_records(self) -> None:
        """从已有 Chroma 集合恢复用于返回检索结果的完整记录。"""
        stored = self._collection.get(include=["documents", "metadatas"])
        ids = stored.get("ids") or []
        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []
        for index, record_id in enumerate(ids):
            metadata = dict(metadatas[index] or {})
            text = str(documents[index] or metadata.get("memory_text", ""))
            metadata["memory_text"] = text
            self.records.append(metadata)
            self._records_by_id[str(record_id)] = metadata

    def _embed(self, texts: list[str]) -> np.ndarray:
        """把文本转换为当前 backend 的归一化向量。"""
        return np.asarray(self._embedder.embed(texts), dtype=np.float32)

    def add(self, record: dict[str, Any]) -> bool:
        """把非空且不重复的经验写入 Chroma 集合。"""
        text = _memory_text(record)
        if not text:
            raise ValueError("memory record has no rule or trigger text")
        fingerprint = _normalise(text)
        if any(_normalise(_memory_text(existing)) == fingerprint for existing in self.records):
            return False
        enriched = dict(record)
        enriched["memory_text"] = text
        row = self._embed([text])
        record_id = f"memory_{uuid.uuid4().hex}"
        # Chroma metadata 只接受标量；当前记录均为标量，仍过滤空值以保证接口稳定。
        metadata = {
            str(key): value
            for key, value in enriched.items()
            if value is not None and isinstance(value, (str, int, float, bool))
        }
        self._collection.add(
            ids=[record_id],
            embeddings=row.tolist(),
            documents=[text],
            metadatas=[metadata],
        )
        self.records.append(enriched)
        self._records_by_id[record_id] = enriched
        return True

    def search(self, query: str, k: int = 2, min_similarity: float = 0.0) -> list[SearchHit]:
        """按余弦相似度返回不低于阈值的前 k 条经验。"""
        if not self.records:
            return []
        row = self._embed([query])
        k = min(k, len(self.records))
        result = self._collection.query(
            query_embeddings=row.tolist(),
            n_results=k,
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits = []
        for record_id, distance in zip(ids, distances):
            # Chroma 的 cosine 返回距离，因此相似度为 1 - distance。
            score = 1.0 - float(distance)
            if score < min_similarity:
                continue
            record = self._records_by_id.get(str(record_id))
            if record is not None:
                hits.append(SearchHit(record, score))
        return hits

    def __len__(self) -> int:
        return len(self.records)

    @property
    def backend(self) -> str:
        """返回当前向量存储后端名称。"""
        return "chromadb"

    @property
    def embedding_name(self) -> str:
        """返回 Chroma 上层的 embedding 实现名称。"""
        return f"{self._embedder.name}:{self._embedder.model_name}"

    @property
    def embedding_dimension(self) -> int:
        """Return the actual backend dimension (384 for MiniLM, for example)."""
        return self._embedder.dimension


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

