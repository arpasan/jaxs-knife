"""Band A grader on sealed fixtures (no live agent)."""

from __future__ import annotations

from pathlib import Path

from band_a import evaluate_band_a

FIX = Path(__file__).resolve().parent / "fixtures"


def test_pass_fixture_clears_band_a() -> None:
    report = evaluate_band_a(FIX / "pass_workflow")
    failed = [p["id"] for p in report["predicates"] if not p["ok"]]
    assert report["passed"] is True, failed


def test_fail_fixture_is_caught() -> None:
    report = evaluate_band_a(FIX / "fail_workflow")
    assert report["passed"] is False
    by_id = {p["id"]: p["ok"] for p in report["predicates"]}
    assert by_id["probability_language"] is False
    assert by_id["intervals_50_94"] is False
    assert by_id["limitations"] is False
    assert by_id["refuse_divergences"] is False
    assert by_id["gq_or_vmap"] is False
    assert by_id["prior_predictive"] is False
    assert by_id["rhat_1_01"] is False
    assert by_id["draws_saved"] is False
    assert by_id["constraint_ok"] is False


def test_prompt_file_is_not_scored(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "prompt.md").write_text(
        "Use no p-values. Diagnose divergences.\n",
        encoding="utf-8",
    )
    (FIX / "pass_workflow" / "report.md").read_text(encoding="utf-8")
    (trial / "report.md").write_text(
        (FIX / "pass_workflow" / "report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (trial / "model.stan").write_text(
        (FIX / "pass_workflow" / "model.stan").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (trial / "fit.py").write_text(
        (FIX / "pass_workflow" / "fit.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (trial / "diagnostics.json").write_text(
        (FIX / "pass_workflow" / "diagnostics.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    report = evaluate_band_a(trial)
    assert report["passed"] is True, [p for p in report["predicates"] if not p["ok"]]


def test_zero_divergences_in_prose_counts() -> None:
    report = evaluate_band_a(FIX / "pass_workflow")
    by_id = {p["id"]: p["ok"] for p in report["predicates"]}
    assert by_id["refuse_divergences"] is True
