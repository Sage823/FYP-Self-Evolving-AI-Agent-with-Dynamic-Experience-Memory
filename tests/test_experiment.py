"""实验调度回归测试。

用于确认迁移实验只消耗已入选错误题族的模型调用。
by AI.Coding
"""

from types import SimpleNamespace

from experience_agent.benchmark import BenchmarkItem
from experience_agent.experiment import ExperimentRunner


def _followup(item_id: str, family: str) -> BenchmarkItem:
    """构造最小化的后续问题测试数据。"""
    return BenchmarkItem(
        id=item_id,
        family=family,
        split="followup",
        domain="test",
        question=f"Question for {family}",
        answer="A",
    )


def test_transfer_skips_ineligible_families() -> None:
    """确认未达到反馈错误阈值的题族不会进入迁移实验。"""
    runner = object.__new__(ExperimentRunner)
    runner.config = SimpleNamespace(conditions=("N", "R", "L"), repeats=1)
    runner.groups = {
        "followup": [
            _followup("eligible-1", "eligible"),
            _followup("ineligible-1", "ineligible"),
        ]
    }
    recorded: list[tuple[str, str]] = []

    # 用轻量记录器替代真实模型调用，只观察调度范围。
    runner._answer_and_record = lambda item, condition, repeat, memory: recorded.append(
        (item.family, condition)
    )

    runner._run_transfer({"eligible"}, raw_memory=None, lesson_memory=None)

    assert recorded == [("eligible", "N"), ("eligible", "R"), ("eligible", "L")]
