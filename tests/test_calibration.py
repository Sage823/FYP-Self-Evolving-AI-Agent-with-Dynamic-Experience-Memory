from pathlib import Path

from experience_agent.calibration import calibrate


ROOT = Path(__file__).resolve().parents[1]


def test_hashing_calibration_uses_validation_and_never_followup(tmp_path: Path) -> None:
    result = calibrate(
        benchmark_path=ROOT / "data" / "benchmark_bbh_v3.jsonl",
        output_path=tmp_path / "calibration.json",
        backends=["hashing"],
        threshold_step=0.1,
    )

    assert result["validation_split"] == "validation"
    assert result["memory_source_split"] == "feedback"
    assert result["followup_used"] is False
    assert "hashing" in result["backends"]
    recommendations = result["backends"]["hashing"]["memory_kinds"]
    assert set(recommendations) == {"raw_correction", "abstract_lesson"}
    assert (tmp_path / "calibration.json").exists()
