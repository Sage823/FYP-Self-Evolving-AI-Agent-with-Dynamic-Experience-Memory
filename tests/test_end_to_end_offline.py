"""离线端到端实验回归测试。

by AI.Coding
"""

from pathlib import Path

from experience_agent.experiment import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]


def test_offline_pipeline_completes(tmp_path: Path) -> None:
    """确认完整流程结束并把两类经验写入 ChromaDB。"""
    runner = ExperimentRunner(
        root=ROOT,
        config_path="config/experiment_v0.json",
        output_dir=tmp_path / "run",
        offline=True,
    )
    summary = runner.run()
    assert summary["status"] == "completed"
    assert summary["eligible_families"]
    assert set(summary["conditions"]) == {"N", "R", "L"}
    assert summary["memory_backend"] == "chromadb"
    assert (tmp_path / "run" / "chroma").is_dir()
    assert (tmp_path / "run" / "events.jsonl").exists()
    assert (tmp_path / "run" / "per_item.csv").exists()
    assert (tmp_path / "run" / "report.md").exists()

