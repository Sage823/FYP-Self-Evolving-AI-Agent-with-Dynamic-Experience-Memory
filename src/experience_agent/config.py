"""实验配置加载与校验。

by AI.Coding
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class ExperimentConfig:
    config_id: str
    model: str
    base_url: str
    temperature: float
    reasoning_effort: str | None
    answer_max_tokens: int
    lesson_max_tokens: int
    repeats: int
    top_k: int
    min_similarity: float
    raw_top_k: int
    raw_min_similarity: float
    lesson_top_k: int
    lesson_min_similarity: float
    embedding_dim: int
    retrieval_backend: str
    embedding_model: str
    max_new_calls: int
    max_total_tokens: int
    feedback_error_threshold: float
    conditions: tuple[str, ...]
    benchmark: str

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """加载项目本地环境变量和 JSON 实验配置。"""
        path = Path(path)
        # 显式环境变量优先，本地 .env 只补充缺失值且始终由 Git 忽略。
        load_dotenv(dotenv_path=path.parent.parent / ".env", override=False)
        raw = json.loads(path.read_text(encoding="utf-8"))
        model = os.environ.get("EXPERIENCE_LLM_MODEL", raw.get("model", "")).strip()
        base_url = os.environ.get("EXPERIENCE_LLM_BASE_URL", raw.get("base_url", "")).strip()
        max_calls = int(os.environ.get("EXPERIENCE_MAX_NEW_CALLS", raw.get("max_new_calls", 120)))
        max_tokens = int(
            os.environ.get("EXPERIENCE_MAX_TOTAL_TOKENS", raw.get("max_total_tokens", 300_000))
        )
        cfg = cls(
            config_id=str(raw["config_id"]),
            model=model,
            base_url=base_url,
            temperature=float(raw.get("temperature", 0.0)),
            reasoning_effort=(
                str(raw["reasoning_effort"]).strip()
                if raw.get("reasoning_effort") is not None
                else None
            ),
            answer_max_tokens=int(raw.get("answer_max_tokens", 220)),
            lesson_max_tokens=int(raw.get("lesson_max_tokens", 320)),
            repeats=int(raw.get("repeats", 1)),
            top_k=int(raw.get("top_k", 2)),
            min_similarity=float(raw.get("min_similarity", 0.08)),
            raw_top_k=int(raw.get("raw_top_k", raw.get("top_k", 2))),
            raw_min_similarity=float(
                raw.get("raw_min_similarity", raw.get("min_similarity", 0.08))
            ),
            lesson_top_k=int(raw.get("lesson_top_k", raw.get("top_k", 2))),
            lesson_min_similarity=float(
                raw.get("lesson_min_similarity", raw.get("min_similarity", 0.08))
            ),
            embedding_dim=int(raw.get("embedding_dim", 2048)),
            retrieval_backend=str(raw.get("retrieval_backend", "hashing")),
            embedding_model=str(
                raw.get(
                    "embedding_model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )
            ),
            max_new_calls=max_calls,
            max_total_tokens=max_tokens,
            feedback_error_threshold=float(raw.get("feedback_error_threshold", 0.3)),
            conditions=tuple(raw.get("conditions", ["N", "R", "L"])),
            benchmark=str(raw.get("benchmark", "data/benchmark_v0.jsonl")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """拒绝会让实验行为含糊或越过预算边界的配置。"""
        if not self.model:
            raise ValueError("model must be non-empty")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        if self.reasoning_effort not in {None, "none", "low", "high", "max"}:
            raise ValueError("reasoning_effort must be none, low, high, max, or null")
        if self.top_k < 1 or self.embedding_dim < 64:
            raise ValueError("top_k must be >= 1 and embedding_dim >= 64")
        if min(self.raw_top_k, self.lesson_top_k) < 1:
            raise ValueError("raw_top_k and lesson_top_k must be >= 1")
        if self.retrieval_backend not in {"hashing", "sentence_transformers"}:
            raise ValueError(
                "retrieval_backend must be 'hashing' or 'sentence_transformers'"
            )
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0 and 1")
        if not 0.0 <= self.raw_min_similarity <= 1.0:
            raise ValueError("raw_min_similarity must be between 0 and 1")
        if not 0.0 <= self.lesson_min_similarity <= 1.0:
            raise ValueError("lesson_min_similarity must be between 0 and 1")
        if not set(self.conditions) <= {"N", "R", "L"}:
            raise ValueError("conditions must be a subset of N, R, L")
        if not 0.0 < self.feedback_error_threshold <= 1.0:
            raise ValueError("feedback_error_threshold must be in (0, 1]")
        if self.max_new_calls < 1 or self.max_total_tokens < 1:
            raise ValueError("call and token budgets must be positive")

