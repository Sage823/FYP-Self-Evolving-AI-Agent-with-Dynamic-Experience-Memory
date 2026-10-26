"""单模型 v0 实验的描述性统计分析。

by AI.Coding
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import binomtest


def _rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(not row["correct"] for row in rows) / len(rows)


def exact_mcnemar(a_wrong: list[bool], b_wrong: list[bool]) -> dict[str, Any]:
    """根据两组配对的错误标记计算精确 McNemar 检验。"""
    if len(a_wrong) != len(b_wrong):
        raise ValueError("paired outcomes differ in length")
    b = sum(left and not right for left, right in zip(a_wrong, b_wrong))
    c = sum(right and not left for left, right in zip(a_wrong, b_wrong))
    if b + c == 0:
        p_value = 1.0
    else:
        # SciPy 默认返回双侧精确 p 值；再次乘二会系统性放大结果。
        p_value = float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"discordant_a_wrong": b, "discordant_b_wrong": c, "p_value": p_value}


def analyse(rows: list[dict[str, Any]], eligible_families: set[str]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("cannot analyse an empty run")
    follow = frame[frame["split"] == "followup"].copy()
    follow = follow[follow["family"].isin(eligible_families)]
    safety = frame[frame["split"] == "unrelated"].copy()

    by_condition: dict[str, Any] = {}
    for condition in sorted(follow["condition"].unique()):
        subset = follow[follow["condition"] == condition]
        by_condition[condition] = {
            "items": int(len(subset)),
            "repeated_error_rate": _rate(subset.to_dict("records")),
            "accuracy": float(1 - _rate(subset.to_dict("records"))) if len(subset) else None,
            "retrieval_hit_rate": (
                float(subset["retrieval_hit"].mean()) if len(subset) else None
            ),
            "mean_retrieved": float(subset["retrieved_count"].mean()) if len(subset) else None,
        }

    by_family: dict[str, Any] = {}
    for family, group in follow.groupby("family"):
        by_family[family] = {
            condition: {
                "items": int(len(part)),
                "repeated_error_rate": _rate(part.to_dict("records")),
                "retrieval_hit_rate": (
                    float(part["retrieval_hit"].mean()) if len(part) else None
                ),
            }
            for condition in sorted(group["condition"].unique())
            for part in [group[group["condition"] == condition]]
        }

    paired: dict[str, Any] = {}
    keys = ["repeat", "item_id"]
    for left, right in (("L", "N"), ("R", "N"), ("L", "R")):
        left_rows = follow[follow["condition"] == left].set_index(keys).sort_index()
        right_rows = follow[follow["condition"] == right].set_index(keys).sort_index()
        common = left_rows.index.intersection(right_rows.index)
        if len(common) == 0:
            continue
        a_wrong = (~left_rows.loc[common, "correct"].astype(bool)).tolist()
        b_wrong = (~right_rows.loc[common, "correct"].astype(bool)).tolist()
        left_rate = sum(a_wrong) / len(common)
        right_rate = sum(b_wrong) / len(common)
        paired[f"{left}_vs_{right}"] = {
            "left_repeated_error_rate": left_rate,
            "right_repeated_error_rate": right_rate,
            "risk_difference_pp": (left_rate - right_rate) * 100,
            "relative_reduction_pct": (
                (right_rate - left_rate) / right_rate * 100 if right_rate else None
            ),
            **exact_mcnemar(a_wrong, b_wrong),
        }

    safety_summary: dict[str, Any] = {}
    for condition in sorted(safety["condition"].unique()):
        subset = safety[safety["condition"] == condition]
        safety_summary[condition] = {
            "items": int(len(subset)),
            "accuracy": float(subset["correct"].mean()),
            "any_retrieval_rate": float((subset["retrieved_count"] > 0).mean()),
        }

    safety_drop = None
    if "N" in safety_summary and "L" in safety_summary:
        safety_drop = (
            safety_summary["N"]["accuracy"] - safety_summary["L"]["accuracy"]
        ) * 100

    h1_reduction = paired.get("L_vs_N", {}).get("relative_reduction_pct")
    h2_reduction = paired.get("L_vs_R", {}).get("relative_reduction_pct")
    return {
        "eligible_families": sorted(eligible_families),
        "conditions": by_condition,
        "by_family": by_family,
        "paired_contrasts": paired,
        "safety": {**safety_summary, "N_minus_L_accuracy_drop_pp": safety_drop},
        "pilot_criteria_met": {
            "H1_at_least_30pct_relative_reduction": (
                h1_reduction is not None and h1_reduction >= 30
            ),
            "H2_L_beats_R": h2_reduction is not None and h2_reduction > 0,
            "unrelated_accuracy_drop_at_most_5pp": (
                safety_drop is not None and safety_drop <= 5
            ),
            "note": "These are descriptive pilot criteria, not a frozen final-study claim.",
        },
    }


def write_analysis(run_dir: str | Path, rows: list[dict[str, Any]], eligible: set[str]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary = analyse(rows, eligible)
    (run_dir / "per_item.csv").write_text(
        pd.DataFrame(rows).to_csv(index=False), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Experience-memory run report",
        "",
        f"Eligible error families: {', '.join(summary['eligible_families']) or 'none'}",
        "",
        "## Repeated error by condition",
        "",
        "| Condition | Items | Repeated error | Accuracy | Retrieval hit rate |",
        "|---|---:|---:|---:|---:|",
    ]
    description = {
        "N": "No memory",
        "R": "Raw correction",
        "L": "Abstracted lesson",
    }
    for condition, value in summary["conditions"].items():
        lines.append(
            f"| {condition} ({description.get(condition, condition)}) | {value['items']} | "
            f"{_fmt_pct(value['repeated_error_rate'])} | {_fmt_pct(value['accuracy'])} | "
            f"{_fmt_pct(value['retrieval_hit_rate'])} |"
        )
    lines += [
        "",
        "## Paired contrasts",
        "",
        "| Contrast | Risk difference (pp) | Relative reduction | Exact McNemar p |",
        "|---|---:|---:|---:|",
    ]
    for name, value in summary["paired_contrasts"].items():
        lines.append(
            f"| {name} | {value['risk_difference_pp']:+.1f} | "
            f"{_fmt_percent_value(value['relative_reduction_pct'])} | {value['p_value']:.4g} |"
        )
    safety = summary["safety"]
    lines += [
        "",
        "## Safety check",
        "",
        f"Accuracy drop from N to L: {_fmt_pp(safety.get('N_minus_L_accuracy_drop_pp'))}",
        "",
        "## Interpretation boundary",
        "",
        "This is a one-model, one-repeat pilot. It verifies the learning loop and produces",
        "descriptive evidence only. Statistical significance and generalisation require a",
        "larger frozen benchmark and multiple repeats.",
    ]
    return "\n".join(lines) + "\n"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_pp(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f} pp"


def _fmt_percent_value(value: float | None) -> str:
    """格式化已经采用百分数单位的数值，不再额外乘以一百。"""
    return "n/a" if value is None else f"{value:.1f}%"

