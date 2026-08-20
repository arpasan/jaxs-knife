"""Band A grader on sealed fixtures (no live agent)."""

from __future__ import annotations

from pathlib import Path

from band_a import evaluate_band_a

FIX = Path(__file__).resolve().parent / "fixtures"


def test_pass_fixture_clears_band_a() -> None:
    report = evaluate_band_a(FIX / "pass_workflow")
    failed = [p["id"] for p in report["predicates"] if not p["ok"]]
    assert report["passed"] is True, failed


def test_pymc_style_ppc_and_halfnormal_count(tmp_path: Path) -> None:
    trial = tmp_path / "pymc"
    trial.mkdir()
    (trial / "report.md").write_text(
        (FIX / "pass_workflow" / "report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (trial / "fit.py").write_text(
        "prior predictive before sampling\n"
        "sigma = pm.HalfNormal('sigma', 1.0)\n"
        "idata = pm.sample()\n"
        "pm.sample_posterior_predictive(idata)\n"
        "idata.to_netcdf('inference_data.nc')\n",
        encoding="utf-8",
    )
    report = evaluate_band_a(trial)
    by_id = {p["id"]: p["ok"] for p in report["predicates"]}
    assert by_id["gq_or_vmap"] is True
    assert by_id["constraint_ok"] is True
    assert report["passed"] is True, [p for p in report["predicates"] if not p["ok"]]


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


def test_prior_sensitivity_refit_is_opt_in(tmp_path: Path) -> None:
    trial = tmp_path / "sense"
    trial.mkdir()
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
    base = evaluate_band_a(trial)
    assert base["passed"] is True
    extra = evaluate_band_a(trial, extra=["prior_sensitivity_refit"])
    by_id = {p["id"]: p["ok"] for p in extra["predicates"]}
    assert extra["passed"] is False
    assert by_id["prior_sensitivity_refit"] is False
    (trial / "report.md").write_text(
        (trial / "report.md").read_text(encoding="utf-8")
        + "\nPrior sensitivity: we refit under a tighter slope prior. "
        "The decision-relevant conclusion does not move.\n",
        encoding="utf-8",
    )
    ok = evaluate_band_a(trial, extra=["prior_sensitivity_refit"])
    assert ok["passed"] is True


def test_copied_skill_does_not_grade_itself() -> None:
    """On-skill folders copy SKILL.md; those files must not satisfy Band A."""
    import shutil

    from workspace import prepare_workspace

    dest = Path(__file__).resolve().parent / "local_runs" / "_pytest" / "contam"
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    junk = "# Report\n\n50% HDI and 94% HDI.\n\nR-hat 1.01.\nZero divergences.\nLimitations: toy data.\n"
    try:
        prepare_workspace("E1", dest, condition="with")
        for path in list(dest.iterdir()):
            if path.name == ".cursor":
                continue
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
        (dest / "report.md").write_text(junk, encoding="utf-8")
        on = evaluate_band_a(dest)
        off_dir = dest.parent / "contam-off"
        if off_dir.exists():
            shutil.rmtree(off_dir)
        off_dir.mkdir()
        (off_dir / "report.md").write_text(junk, encoding="utf-8")
        off = evaluate_band_a(off_dir)
        leaked = ("prior_predictive", "gq_or_vmap", "constraint_ok", "draws_saved")
        on_ids = {p["id"]: p["ok"] for p in on["predicates"]}
        off_ids = {p["id"]: p["ok"] for p in off["predicates"]}
        for pid in leaked:
            assert on_ids[pid] is False, pid
            assert off_ids[pid] is False, pid
        assert on["n_pass"] == off["n_pass"]
    finally:
        shutil.rmtree(dest, ignore_errors=True)
        shutil.rmtree(dest.parent / "contam-off", ignore_errors=True)
