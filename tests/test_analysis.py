"""统计分析回归测试。

用于锁定精确 McNemar 检验的双侧 p 值计算，避免重复加倍。
by AI.Coding
"""

import pytest

from experience_agent.analysis import analyse, exact_mcnemar, render_report


def test_exact_mcnemar_uses_two_sided_p_value_once() -> None:
    """确认 SciPy 已返回双侧 p 值时不再额外乘以二。"""
    # 构造 1 比 9 的不一致配对；二项分布双侧精确 p 值为 0.021484375。
    left_wrong = [True] + [False] * 9
    right_wrong = [False] + [True] * 9

    result = exact_mcnemar(left_wrong, right_wrong)

    assert result["p_value"] == pytest.approx(0.021484375)


def test_safety_drop_name_matches_n_minus_l_direction() -> None:
    """确认安全性下降指标的名称与实际计算方向一致。"""
    # N 条件答对而 L 条件答错时，准确率下降应记为正 100 个百分点。
    rows = [
        {
            "item_id": "u1",
            "family": "unrelated",
            "split": "unrelated",
            "condition": "N",
            "repeat": 0,
            "correct": True,
            "retrieved_count": 0,
            "retrieval_hit": False,
        },
        {
            "item_id": "u1",
            "family": "unrelated",
            "split": "unrelated",
            "condition": "L",
            "repeat": 0,
            "correct": False,
            "retrieved_count": 1,
            "retrieval_hit": False,
        },
    ]

    summary = analyse(rows, eligible_families=set())

    assert summary["safety"]["N_minus_L_accuracy_drop_pp"] == 100.0
    assert "Accuracy drop from N to L: +100.0 pp" in render_report(summary)


def test_report_does_not_multiply_relative_reduction_twice() -> None:
    """确认已经采用百分数单位的相对下降不会再次乘以一百。"""
    summary = {
        "eligible_families": ["test"],
        "conditions": {},
        "paired_contrasts": {
            "R_vs_N": {
                "risk_difference_pp": -18.75,
                "relative_reduction_pct": 33.3333333333,
                "p_value": 0.25,
            }
        },
        "safety": {"N_minus_L_accuracy_drop_pp": 0.0},
    }

    report = render_report(summary)

    assert "33.3%" in report
    assert "3333.3%" not in report


def test_incomplete_answers_are_excluded_from_repeated_error_rate() -> None:
    base = {
        "item_id": "q1",
        "family": "f",
        "split": "followup",
        "repeat": 0,
        "retrieved_count": 0,
        "retrieval_hit": False,
        "answer_valid": True,
    }
    rows = [
        {**base, "condition": "N", "correct": False, "status": "COMPLETE_WRONG"},
        {**base, "condition": "R", "correct": True, "status": "COMPLETE_CORRECT"},
        {
            **base,
            "condition": "L",
            "correct": False,
            "answer_valid": False,
            "status": "TRUNCATED",
        },
    ]
    summary = analyse(rows, eligible_families={"f"})
    assert summary["conditions"]["L"]["items"] == 0
    assert summary["conditions"]["L"]["excluded_items"] == 1
    assert summary["excluded_answers"] == 1
    assert summary["paired_contrasts"]["R_vs_N"]["left_repeated_error_rate"] == 0.0
