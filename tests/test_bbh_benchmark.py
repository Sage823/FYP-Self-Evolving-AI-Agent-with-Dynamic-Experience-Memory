"""BBH 风格 v1 benchmark 的生成与结构回归测试。

by AI.Coding
"""

from pathlib import Path

from experience_agent.benchmark import benchmark_stats, load_benchmark, validate_benchmark
from experience_agent.bbh_benchmark import build_benchmark, build_expanded_followups


def test_bbh_v1_is_deterministic_and_balanced() -> None:
    """确认固定种子生成稳定、有效且分组平衡的 32 道题。"""
    first = build_benchmark(seed=20261026)
    second = build_benchmark(seed=20261026)

    assert first == second
    validate_benchmark(first)
    stats = benchmark_stats(first)
    assert stats["total"] == 32
    assert stats["by_split"] == {"feedback": 8, "followup": 16, "unrelated": 8}
    assert stats["by_family"] == {
        "B1_state_tracking": 6,
        "B2_boolean_logic": 6,
        "B3_date_arithmetic": 6,
        "B4_dyck_completion": 6,
    }


def test_feedback_items_have_reusable_corrections() -> None:
    """确认每道反馈题都带有答案与可迁移的纠错原则。"""
    feedback = [item for item in build_benchmark() if item.split == "feedback"]

    assert len(feedback) == 8
    assert all(item.correction for item in feedback)
    assert all("General method:" in item.correction for item in feedback)


def test_bbh_v2_is_valid_and_materially_longer() -> None:
    """确认 v2 保持相同结构，同时显著提高反馈题文本长度。"""
    v1 = build_benchmark(difficulty=1)
    v2 = build_benchmark(difficulty=2)

    validate_benchmark(v2)
    v1_feedback_chars = sum(len(item.question) for item in v1 if item.split == "feedback")
    v2_feedback_chars = sum(len(item.question) for item in v2 if item.split == "feedback")
    assert v2_feedback_chars > v1_feedback_chars * 1.7


def test_calibrated_benchmark_has_independent_development_splits() -> None:
    """Baseline selects families, validation calibrates retrieval, follow-up evaluates."""
    items = build_benchmark(difficulty=2, with_development_splits=True)
    validate_benchmark(items)
    stats = benchmark_stats(items)

    assert stats["total"] == 48
    assert stats["by_split"] == {
        "baseline": 8,
        "feedback": 8,
        "followup": 16,
        "unrelated": 8,
        "validation": 8,
    }


def test_expanded_followups_keep_development_splits_unchanged() -> None:
    base = build_benchmark(difficulty=2, with_development_splits=True)
    expanded = build_expanded_followups(
        seed=20261026,
        difficulty=2,
        followup_per_family=20,
    )
    base_development = [item for item in base if item.split != "followup"]
    expanded_development = [item for item in expanded if item.split != "followup"]

    assert base_development == expanded_development
    stats = benchmark_stats(expanded)
    assert stats["total"] == 112
    assert stats["by_split"]["followup"] == 80
    assert stats["by_split"]["baseline"] == 8
    assert stats["by_split"]["feedback"] == 8
    assert stats["by_split"]["validation"] == 8
    assert stats["by_split"]["unrelated"] == 8


def test_expanded_followups_can_freeze_an_existing_file() -> None:
    root = Path(__file__).resolve().parents[1]
    base_path = root / "data" / "benchmark_bbh_v3.jsonl"
    base = load_benchmark(base_path)
    expanded = build_expanded_followups(
        seed=20261026,
        difficulty=2,
        followup_per_family=20,
        base_benchmark=base_path,
    )
    assert [item for item in base if item.split != "followup"] == [
        item for item in expanded if item.split != "followup"
    ]
