"""BBH 风格 v1 benchmark 的生成与结构回归测试。

by AI.Coding
"""

from experience_agent.benchmark import benchmark_stats, validate_benchmark
from experience_agent.bbh_benchmark import build_benchmark


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
