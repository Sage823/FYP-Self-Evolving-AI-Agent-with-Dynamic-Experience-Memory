from pathlib import Path

from experience_agent.benchmark import benchmark_stats, load_benchmark, normalize_answer


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_is_valid_and_balanced() -> None:
    items = load_benchmark(ROOT / "data" / "benchmark_v0.jsonl")
    stats = benchmark_stats(items)
    assert stats["total"] == 32
    assert stats["by_split"] == {"feedback": 8, "followup": 16, "unrelated": 8}
    assert all(count == 6 for count in stats["by_family"].values())


def test_answer_normalisation() -> None:
    assert normalize_answer(" Answer: No. ") == "NO"
    assert normalize_answer("D") == "D"

