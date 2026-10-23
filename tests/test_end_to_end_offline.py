from pathlib import Path

from experience_agent.experiment import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]


def test_offline_pipeline_completes(tmp_path: Path) -> None:
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
    assert (tmp_path / "run" / "events.jsonl").exists()
    assert (tmp_path / "run" / "per_item.csv").exists()
    assert (tmp_path / "run" / "report.md").exists()

