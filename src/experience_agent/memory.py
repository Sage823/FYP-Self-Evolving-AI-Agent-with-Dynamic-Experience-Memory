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
from sklearn.feature_extraction.text import HashingVectorizer


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
    """用 Chroma 集合保存经验，并用本地哈希向量执行检索。"""

    dimension: int = 2048
    persist_directory: str | Path | None = None
    collection_name: str = "experience_memory"
    records: list[dict[str, Any]] = field(default_factory=list)
    _client: Any = field(default=None, init=False, repr=False)
    _collection: Any = field(default=None, init=False, repr=False)
    _records_by_id: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """建立 Chroma 客户端、集合和本地向量器，并恢复已有记录。"""
        self._vectorizer = HashingVectorizer(
            n_features=self.dimension,
            alternate_sign=False,
            norm="l2",
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
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
        )
        self._load_existing_records()

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
        """把文本转换为归一化、可复现且无需下载模型的稠密向量。"""
        matrix = self._vectorizer.transform(texts).astype(np.float32)
        embeddings = matrix.toarray()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

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


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

