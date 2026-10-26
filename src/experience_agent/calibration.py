"""Compare embedding backends and calibrate retrieval on validation data.

The calibration uses:
* feedback items to build the memory;
* the separate validation split as positive retrieval queries;
* unrelated items as negative queries.

Follow-up items are never read by this module. This prevents threshold selection
from leaking test-set information into the main experiment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkItem, load_benchmark, split_items
from .memory import ExperienceMemory, SearchHit


DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def abstract_lesson_proxy(item: BenchmarkItem) -> str:
    """Extract the reusable method from a generated correction without an LLM."""
    correction = (item.correction or "").strip()
    match = re.search(r"General method:\s*(.+)$", correction, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else correction


def memory_records(feedback: list[BenchmarkItem], kind: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in feedback:
        if not item.correction:
            continue
        records.append(
            {
                "kind": kind,
                "error_type": item.family,
                "source_id": item.id,
                "rule": (
                    item.correction
                    if kind == "raw_correction"
                    else abstract_lesson_proxy(item)
                ),
                "trigger": "",
            }
        )
    return records


def _ranked_hits(
    memory: ExperienceMemory,
    queries: list[BenchmarkItem],
    *,
    max_k: int,
) -> dict[str, list[SearchHit]]:
    return {
        item.id: memory.search(item.question, k=max_k, min_similarity=-1.0)
        for item in queries
    }


def _metrics_at(
    *,
    positives: list[BenchmarkItem],
    negatives: list[BenchmarkItem],
    positive_hits: dict[str, list[SearchHit]],
    negative_hits: dict[str, list[SearchHit]],
    threshold: float,
    top_k: int,
) -> dict[str, float | int]:
    hit_count = 0
    top1_family_count = 0
    relevant_found = 0
    total_found = 0
    for item in positives:
        selected = [
            hit
            for hit in positive_hits[item.id][:top_k]
            if hit.score >= threshold
        ]
        total_found += len(selected)
        relevant_found += sum(
            hit.record.get("error_type") == item.family for hit in selected
        )
        hit_count += any(
            hit.record.get("error_type") == item.family for hit in selected
        )
        top1_family_count += bool(
            selected and selected[0].record.get("error_type") == item.family
        )
    false_positives = 0
    for item in negatives:
        selected = [
            hit
            for hit in negative_hits[item.id][:top_k]
            if hit.score >= threshold
        ]
        false_positives += bool(selected)
    positive_count = max(1, len(positives))
    negative_count = max(1, len(negatives))
    return {
        "threshold": round(threshold, 4),
        "top_k": top_k,
        "positive_queries": len(positives),
        "hit_count": hit_count,
        "hit_rate": hit_count / positive_count,
        "top1_same_family_rate": top1_family_count / positive_count,
        "precision": relevant_found / total_found if total_found else 0.0,
        "negative_queries": len(negatives),
        "false_positives": false_positives,
        "false_positive_rate": false_positives / negative_count,
    }


def _recommend(
    candidates: list[dict[str, float | int]],
    *,
    max_false_positive_rate: float,
) -> dict[str, float | int]:
    allowed = [
        row
        for row in candidates
        if float(row["false_positive_rate"]) <= max_false_positive_rate
    ]
    pool = allowed or candidates
    return max(
        pool,
        key=lambda row: (
            float(row["hit_rate"]),
            float(row["precision"]),
            float(row["top1_same_family_rate"]),
            -float(row["false_positive_rate"]),
            -int(row["top_k"]),
            float(row["threshold"]),
        ),
    )


def calibrate_backend(
    *,
    backend: str,
    embedding_model: str,
    dimension: int,
    development: dict[str, list[BenchmarkItem]],
    top_k_values: list[int],
    thresholds: list[float],
    max_false_positive_rate: float,
) -> dict[str, Any]:
    feedback = development["feedback"]
    positives = development["validation"]
    negatives = development["unrelated"]
    max_k = max(top_k_values)
    by_memory_kind: dict[str, Any] = {}

    for kind in ("raw_correction", "abstract_lesson"):
        memory = ExperienceMemory(
            dimension=dimension,
            embedding_backend=backend,
            embedding_model=embedding_model,
        )
        for record in memory_records(feedback, kind):
            memory.add(record)
        positive_hits = _ranked_hits(memory, positives, max_k=max_k)
        negative_hits = _ranked_hits(memory, negatives, max_k=max_k)
        candidates = [
            {
                "memory_kind": kind,
                "backend": backend,
                **_metrics_at(
                    positives=positives,
                    negatives=negatives,
                    positive_hits=positive_hits,
                    negative_hits=negative_hits,
                    threshold=threshold,
                    top_k=top_k,
                ),
            }
            for top_k in top_k_values
            for threshold in thresholds
        ]
        recommendation = _recommend(
            candidates,
            max_false_positive_rate=max_false_positive_rate,
        )
        by_memory_kind[kind] = {
            "memory_records": len(memory),
            "embedding_dimension": memory.embedding_dimension,
            "recommendation": recommendation,
            "candidates": candidates,
        }
    return {"backend": backend, "memory_kinds": by_memory_kind}


def calibrate(
    *,
    benchmark_path: str | Path,
    output_path: str | Path,
    backends: list[str],
    semantic_model: str = DEFAULT_SEMANTIC_MODEL,
    dimension: int = 2048,
    top_k_values: list[int] | None = None,
    threshold_step: float = 0.05,
    max_false_positive_rate: float = 0.2,
) -> dict[str, Any]:
    items = load_benchmark(benchmark_path)
    development = split_items(items)
    if "validation" not in development:
        raise ValueError(
            "retrieval calibration requires an independent 'validation' split; "
            "generate the benchmark with --with-development-splits"
        )
    top_k_values = sorted(set(top_k_values or [1, 2, 3, 5]))
    if threshold_step <= 0 or threshold_step > 1:
        raise ValueError("threshold_step must be in (0, 1]")
    thresholds = [round(index * threshold_step, 4) for index in range(int(1 / threshold_step) + 1)]
    result: dict[str, Any] = {
        "benchmark": str(Path(benchmark_path)),
        "validation_split": "validation",
        "memory_source_split": "feedback",
        "followup_used": False,
        "semantic_model": semantic_model,
        "top_k_values": top_k_values,
        "thresholds": thresholds,
        "max_false_positive_rate": max_false_positive_rate,
        "backends": {},
        "backend_errors": {},
    }
    for backend in backends:
        try:
            result["backends"][backend] = calibrate_backend(
                backend=backend,
                embedding_model=semantic_model,
                dimension=dimension,
                development=development,
                top_k_values=top_k_values,
                thresholds=thresholds,
                max_false_positive_rate=max_false_positive_rate,
            )
        except Exception as exc:  # Optional model download/install can fail independently.
            result["backend_errors"][backend] = f"{type(exc).__name__}: {exc}"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def render_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Retrieval calibration",
        "",
        "Selection uses feedback memories, validation positives, and unrelated negatives.",
        "Follow-up items are not used during calibration.",
        "",
        "| Backend | Memory | Hit rate | P@k | Top-1 family | FPR | top-k | threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend, payload in result["backends"].items():
        for kind, details in payload["memory_kinds"].items():
            row = details["recommendation"]
            lines.append(
                f"| {backend} | {kind} | {row['hit_rate']:.3f} | "
                f"{row['precision']:.3f} | {row['top1_same_family_rate']:.3f} | "
                f"{row['false_positive_rate']:.3f} | {row['top_k']} | {row['threshold']:.2f} |"
            )
    for backend, error in result["backend_errors"].items():
        lines += ["", f"Backend `{backend}` was not evaluated: {error}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare and calibrate memory retrieval")
    parser.add_argument("--benchmark", default="data/benchmark_bbh_v3.jsonl")
    parser.add_argument(
        "--output",
        default="results/calibration/retrieval_calibration.json",
    )
    parser.add_argument(
        "--backends",
        default="hashing,sentence_transformers",
        help="comma-separated: hashing,sentence_transformers",
    )
    parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--top-k", default="1,2,3,5")
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.2)
    args = parser.parse_args(argv)
    result = calibrate(
        benchmark_path=args.benchmark,
        output_path=args.output,
        backends=[value.strip() for value in args.backends.split(",") if value.strip()],
        semantic_model=args.semantic_model,
        dimension=args.embedding_dim,
        top_k_values=[int(value) for value in args.top_k.split(",") if value.strip()],
        threshold_step=args.threshold_step,
        max_false_positive_rate=args.max_false_positive_rate,
    )
    summary_path = Path(args.output).with_suffix(".md")
    summary_path.write_text(render_summary(result), encoding="utf-8")
    print(render_summary(result))
    return 0 if result["backends"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

