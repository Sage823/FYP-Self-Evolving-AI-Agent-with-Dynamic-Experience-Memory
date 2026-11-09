"""实验调度回归测试。

用于确认迁移实验只消耗已入选错误题族的模型调用。
by AI.Coding
"""

import json
from types import SimpleNamespace

from experience_agent.agent import AnswerResult
from experience_agent.benchmark import BenchmarkItem
from experience_agent.experiment import ExperimentRunner
from experience_agent.memory import SearchHit


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


def test_retrieval_trace_keeps_candidates_below_threshold(tmp_path) -> None:
    runner = object.__new__(ExperimentRunner)
    runner.config = SimpleNamespace(
        top_k=2, min_similarity=0.5,
        raw_top_k=2, raw_min_similarity=0.5,
        lesson_top_k=2, lesson_min_similarity=0.5,
    )
    runner.rows = []
    runner.events_path = tmp_path / "events.jsonl"
    supplied_memories = []

    def answer(item, memories):
        supplied_memories.append(memories)
        return AnswerResult(
            raw_text="Answer: A", answer="A", correct=True,
            status="COMPLETE_CORRECT",
        )

    runner.agent = SimpleNamespace(answer=answer)
    candidates = [
        SearchHit(record={"source_id": "low", "error_type": "B1"}, score=0.2),
        SearchHit(record={"source_id": "high", "error_type": "B1"}, score=0.8),
    ]
    search_arguments = []

    def search(question, k, min_similarity):
        search_arguments.append((k, min_similarity))
        return candidates

    item = BenchmarkItem(
        id="trace-1", family="B1", split="followup", domain="test",
        question="Track the swaps", answer="A",
    )
    runner._answer_and_record(item, "R", 0, SimpleNamespace(search=search))
    row = runner.rows[0]
    assert search_arguments == [(2, float("-inf"))]
    assert [record["source_id"] for record in supplied_memories[0]] == ["high"]
    assert json.loads(row["candidate_ids"]) == ["low", "high"]
    assert json.loads(row["selected_ids"]) == ["high"]
    assert row["similarity_threshold"] == 0.5
    assert row["retrieval_status"] == "same_family_proxy_selected"
    assert row["retrieval_hit_basis"] == "same_family_proxy_not_applicability_gold"
    assert json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])["candidate_ids"] == ["low", "high"]

    candidates[:] = [candidates[0]]
    runner._answer_and_record(item, "R", 1, SimpleNamespace(search=search))
    assert runner.rows[1]["retrieval_status"] == "below_threshold"
    assert json.loads(runner.rows[1]["candidate_ids"]) == ["low"]
    assert json.loads(runner.rows[1]["selected_ids"]) == []
    assert supplied_memories[1] == []
