"""Benchmark loading, validation, and deterministic statistics."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPLITS = {"feedback", "followup", "unrelated"}


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    family: str
    split: str
    domain: str
    question: str
    answer: str
    distractor: str | None = None
    correction: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "BenchmarkItem":
        required = ("id", "family", "split", "domain", "question", "answer")
        missing = [key for key in required if not str(raw.get(key, "")).strip()]
        if missing:
            raise ValueError(f"benchmark item missing fields {missing}: {raw}")
        split = str(raw["split"])
        correction = str(raw.get("correction", "")).strip() or None
        if split == "feedback" and not correction:
            raise ValueError(f"feedback item {raw['id']} needs a correction")
        return cls(
            id=str(raw["id"]).strip(),
            family=str(raw["family"]).strip(),
            split=split,
            domain=str(raw["domain"]).strip(),
            question=str(raw["question"]).strip(),
            answer=normalize_answer(str(raw["answer"])),
            distractor=(
                normalize_answer(str(raw["distractor"]))
                if raw.get("distractor") is not None
                else None
            ),
            correction=correction,
        )


def normalize_answer(text: str) -> str:
    """Normalise labels while ignoring a leading 'Answer:' marker."""
    text = text.strip()
    matches = re.findall(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$", text)
    if matches:
        text = matches[-1]
    text = text.strip().strip("\"'`.,;:()[]{} ")
    return text.upper()


def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    path = Path(path)
    items: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            items.append(BenchmarkItem.from_dict(raw))
    validate_benchmark(items)
    return items


def validate_benchmark(items: Iterable[BenchmarkItem]) -> None:
    items = list(items)
    if not items:
        raise ValueError("benchmark is empty")
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate benchmark ids: {duplicates}")
    if any(item.split not in SPLITS for item in items):
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    questions = [re.sub(r"\s+", " ", item.question.strip().lower()) for item in items]
    if len(questions) != len(set(questions)):
        raise ValueError("benchmark contains duplicate question text")

    followup_families = {item.family for item in items if item.split == "followup"}
    feedback_families = {item.family for item in items if item.split == "feedback"}
    if followup_families != feedback_families:
        raise ValueError(
            "feedback and follow-up families differ: "
            f"feedback={sorted(feedback_families)}, followup={sorted(followup_families)}"
        )
    unrelated_families = {item.family for item in items if item.split == "unrelated"}
    if unrelated_families != {"unrelated"}:
        raise ValueError("unrelated split must use family='unrelated' only")
    for split in SPLITS:
        if not any(item.split == split for item in items):
            raise ValueError(f"benchmark has no {split} items")
    for item in items:
        if item.split != "unrelated" and item.distractor is not None:
            if item.distractor == item.answer:
                raise ValueError(f"{item.id}: distractor equals answer")


def split_items(items: Iterable[BenchmarkItem]) -> dict[str, list[BenchmarkItem]]:
    grouped: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for item in items:
        grouped[item.split].append(item)
    return dict(grouped)


def benchmark_stats(items: Iterable[BenchmarkItem]) -> dict[str, object]:
    items = list(items)
    by_split = Counter(item.split for item in items)
    by_family = Counter(item.family for item in items if item.split != "unrelated")
    return {
        "total": len(items),
        "by_split": dict(sorted(by_split.items())),
        "by_family": dict(sorted(by_family.items())),
    }

