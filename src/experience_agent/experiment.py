"""单模型 v0 试验的端到端学习循环调度器。

by AI.Coding
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import ExperienceAgent
from .analysis import write_analysis
from .benchmark import BenchmarkItem, load_benchmark, split_items
from .config import ExperimentConfig
from .llm import LLMClient, MockLLM
from .memory import ExperienceMemory


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class ExperimentRunner:
    def __init__(
        self,
        *,
        root: str | Path,
        config_path: str | Path,
        output_dir: str | Path | None = None,
        offline: bool = False,
    ) -> None:
        """加载配置与题库，并建立真实或离线模型客户端。"""
        self.root = Path(root).resolve()
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = self.root / self.config_path
        self.config = ExperimentConfig.load(self.config_path)
        benchmark_path = Path(self.config.benchmark)
        if not benchmark_path.is_absolute():
            benchmark_path = self.root / benchmark_path
        self.benchmark_path = benchmark_path
        self.items = load_benchmark(benchmark_path)
        self.groups = split_items(self.items)
        self.offline = offline
        if output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_dir = self.root / "results" / "runs" / f"{self.config.config_id}_{stamp}"
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = self.root / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.output_dir / "events.jsonl"
        self.rows: list[dict[str, Any]] = []
        if offline:
            self.client: LLMClient | MockLLM = MockLLM()
        else:
            self.client = LLMClient(
                model=self.config.model,
                base_url=self.config.base_url,
                cache_path=self.root / "results" / "cache" / "llm_cache.jsonl",
                temperature=self.config.temperature,
                reasoning_effort=self.config.reasoning_effort,
                max_new_calls=self.config.max_new_calls,
                max_total_tokens=self.config.max_total_tokens,
            )
        self.agent = ExperienceAgent(self.client, self.config)
        _write_json(self.output_dir / "config_used.json", asdict(self.config))
        (self.output_dir / "benchmark.sha256").write_text(
            _sha256(self.benchmark_path) + "\n", encoding="utf-8"
        )

    def _event(self, event_type: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **fields,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self) -> dict[str, Any]:
        self._event(
            "run_started",
            offline=self.offline,
            model=self.client.model,
            benchmark=str(self.benchmark_path),
        )
        feedback_rows = self._run_feedback_baseline()
        family_errors: dict[str, list[dict[str, Any]]] = {}
        for row in feedback_rows:
            if not row["correct"]:
                family_errors.setdefault(row["family"], []).append(row)
        feedback_counts: dict[str, int] = {}
        for item in self.groups["feedback"]:
            feedback_counts[item.family] = feedback_counts.get(item.family, 0) + 1
        eligible = {
            family
            for family, errors in family_errors.items()
            if len(errors) / feedback_counts[family] >= self.config.feedback_error_threshold
        }
        self._event(
            "family_selection",
            feedback_error_rate={
                family: len(family_errors.get(family, [])) / count
                for family, count in sorted(feedback_counts.items())
            },
            eligible_families=sorted(eligible),
        )
        if not eligible:
            summary = {
                "status": "no_eligible_error_families",
                "feedback_error_rate": {
                    family: len(family_errors.get(family, [])) / count
                    for family, count in sorted(feedback_counts.items())
                },
                "message": (
                    "No family crossed the feedback error threshold. The agent is too strong "
                    "for this benchmark; design harder items before testing memory transfer."
                ),
            }
            _write_json(self.output_dir / "summary.json", summary)
            return summary

        raw_memory, lesson_memory = self._build_memories(eligible, family_errors)
        self._run_transfer(eligible, raw_memory, lesson_memory)
        self._run_safety(raw_memory, lesson_memory)
        summary = write_analysis(self.output_dir, self.rows, eligible)
        summary.update(
            {
                "status": "completed",
                "offline": self.offline,
                "model": self.client.model,
                "memory_records": {"R": len(raw_memory), "L": len(lesson_memory)},
                "memory_backend": raw_memory.backend,
                "usage_all_invocations": self.client.usage.as_dict(),
                "usage_billable": self.client.billable_usage.as_dict(),
                "new_llm_calls": self.client.new_calls,
            }
        )
        _write_json(self.output_dir / "summary.json", summary)
        _write_json(
            self.output_dir / "memories.json",
            {"R": raw_memory.records, "L": lesson_memory.records},
        )
        self._event("run_completed", status="completed", **summary["usage_billable"])
        return summary

    def _run_feedback_baseline(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.groups["feedback"]:
            answer = self.agent.answer(item, memories=[])
            row = self._base_row(item, condition="N", repeat=0)
            row.update(answer.as_dict())
            rows.append(row)
            self.rows.append(row)
            self._event(
                "feedback_baseline",
                item_id=item.id,
                family=item.family,
                predicted=answer.answer,
                gold=item.answer,
                correct=answer.correct,
            )
        return rows

    def _build_memories(
        self,
        eligible: set[str],
        family_errors: dict[str, list[dict[str, Any]]],
    ) -> tuple[ExperienceMemory, ExperienceMemory]:
        """为原始纠正和抽象经验建立相互隔离的持久 Chroma 集合。"""
        chroma_directory = self.output_dir / "chroma"
        raw = ExperienceMemory(
            dimension=self.config.embedding_dim,
            persist_directory=chroma_directory,
            collection_name="raw_corrections",
        )
        lesson = ExperienceMemory(
            dimension=self.config.embedding_dim,
            persist_directory=chroma_directory,
            collection_name="abstract_lessons",
        )
        errors_by_id = {
            row["item_id"]: row
            for family in family_errors.values()
            for row in family
        }
        for item in self.groups["feedback"]:
            if item.family not in eligible or item.id not in errors_by_id:
                continue
            if not item.correction:
                continue
            raw.add(
                {
                    "kind": "raw_correction",
                    "error_type": item.family,
                    "source_id": item.id,
                    "rule": item.correction,
                    "trigger": "",
                }
            )
            try:
                extracted = self.agent.extract_lesson(item, errors_by_id[item.id]["answer"])
            except Exception as exc:
                self._event(
                    "lesson_extraction_failed",
                    item_id=item.id,
                    family=item.family,
                    error=str(exc),
                )
                continue
            lesson.add(
                {
                    "kind": "abstract_lesson",
                    "error_type": item.family,
                    "source_id": item.id,
                    **extracted.as_dict(),
                }
            )
            self._event(
                "lesson_stored",
                item_id=item.id,
                family=item.family,
                lesson=extracted.as_dict(),
            )
        self._event("memory_built", raw_records=len(raw), lesson_records=len(lesson))
        return raw, lesson

    def _run_transfer(
        self,
        eligible: set[str],
        raw_memory: ExperienceMemory,
        lesson_memory: ExperienceMemory,
    ) -> None:
        """只对达到反馈错误阈值的题族执行迁移实验。"""
        conditions = list(self.config.conditions)
        for repeat in range(self.config.repeats):
            # 未入选题族不进入主要迁移分析，跳过它们可避免无效模型调用。
            ordered = [
                item for item in self.groups["followup"] if item.family in eligible
            ]
            random.Random(20261023 + repeat).shuffle(ordered)
            for item in ordered:
                for condition in conditions:
                    memory = self._memory_for(condition, raw_memory, lesson_memory)
                    self._answer_and_record(item, condition, repeat, memory)

    def _run_safety(
        self,
        raw_memory: ExperienceMemory,
        lesson_memory: ExperienceMemory,
    ) -> None:
        conditions = list(self.config.conditions)
        for repeat in range(self.config.repeats):
            for item in self.groups["unrelated"]:
                for condition in conditions:
                    memory = self._memory_for(condition, raw_memory, lesson_memory)
                    self._answer_and_record(item, condition, repeat, memory)

    @staticmethod
    def _memory_for(
        condition: str,
        raw_memory: ExperienceMemory,
        lesson_memory: ExperienceMemory,
    ) -> ExperienceMemory | None:
        if condition == "R":
            return raw_memory
        if condition == "L":
            return lesson_memory
        return None

    def _answer_and_record(
        self,
        item: BenchmarkItem,
        condition: str,
        repeat: int,
        memory: ExperienceMemory | None,
    ) -> None:
        hits = (
            memory.search(
                item.question,
                k=self.config.top_k,
                min_similarity=self.config.min_similarity,
            )
            if memory is not None
            else []
        )
        retrieved = [hit.record for hit in hits]
        answer = self.agent.answer(item, memories=retrieved)
        retrieval_hit = any(record.get("error_type") == item.family for record in retrieved)
        row = self._base_row(item, condition=condition, repeat=repeat)
        row.update(answer.as_dict())
        row.update(
            {
                "retrieved_count": len(retrieved),
                "retrieved_ids": "|".join(
                    str(record.get("source_id", "")) for record in retrieved
                ),
                "retrieval_scores": "|".join(f"{hit.score:.5f}" for hit in hits),
                "retrieval_hit": bool(retrieval_hit),
            }
        )
        self.rows.append(row)
        self._event(
            "answer",
            item_id=item.id,
            family=item.family,
            split=item.split,
            condition=condition,
            repeat=repeat,
            predicted=answer.answer,
            gold=item.answer,
            correct=answer.correct,
            retrieved=[record.get("source_id") for record in retrieved],
            retrieval_scores=[round(hit.score, 5) for hit in hits],
        )

    @staticmethod
    def _base_row(item: BenchmarkItem, *, condition: str, repeat: int) -> dict[str, Any]:
        return {
            "item_id": item.id,
            "family": item.family,
            "split": item.split,
            "domain": item.domain,
            "condition": condition,
            "repeat": repeat,
            "question": item.question,
            "gold": item.answer,
        }

