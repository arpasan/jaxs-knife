"""L1: convergence and calibration ratings from JSON (no ArviZ)."""

from __future__ import annotations

from check_diagnostics import check_diagnostics, suggest_next_steps


# ==================================================
# Fixtures
# ==================================================


def _healthy_diagnostics() -> dict:
    return {
        "convergence": {
            "all_ok": True,
            "method": "manual",
            "rhat": {"ok": True, "max": 1.001, "problematic_params": []},
            "ess_bulk": {"ok": True, "min": 800, "problematic_params": []},
            "ess_tail": {"ok": True, "min": 700, "problematic_params": []},
            "divergences": {"count": 0, "pct": 0.0, "ok": True},
        },
        "loo": {
            "computed": True,
            "pareto_k": {"max": 0.3, "n_bad": 0, "n_marginal": 0, "n_nonfinite": 0, "ok": True},
        },
        "posterior_predictive": {"available": True, "variables": ["y_rep"]},
    }


# ==================================================
# S1a — empty convergence is unknown
# ==================================================


def test_empty_convergence_is_unknown_not_excellent() -> None:
    diag = _healthy_diagnostics()
    diag["convergence"] = {}
    report = check_diagnostics(diagnostics=diag)
    assert report["convergence"]["rating"] == "unknown"
    assert report["convergence"]["rating"] != "excellent"
    text = report["summary"]["convergence"].lower()
    assert "unknown" in text
    assert "all convergence diagnostics passed" not in text
    steps = suggest_next_steps(report)
    blob = " ".join(steps).lower()
    assert "acceptable bounds" not in blob
    assert "unknown" in blob


def test_missing_convergence_is_unknown() -> None:
    diag = _healthy_diagnostics()
    del diag["convergence"]
    report = check_diagnostics(diagnostics=diag)
    assert report["convergence"]["rating"] == "unknown"
    assert report["convergence"]["rating"] != "excellent"


def test_none_convergence_is_unknown() -> None:
    diag = _healthy_diagnostics()
    diag["convergence"] = None
    report = check_diagnostics(diagnostics=diag)
    assert report["convergence"]["rating"] == "unknown"


def test_all_ok_still_excellent() -> None:
    report = check_diagnostics(diagnostics=_healthy_diagnostics())
    assert report["convergence"]["rating"] == "excellent"
    steps = suggest_next_steps(report)
    assert any("acceptable bounds" in s for s in steps)


# ==================================================
# S2 — over-coverage is not a prior-tuning failure
# ==================================================


def test_overcoverage_is_not_poor_and_not_tighten_priors() -> None:
    cal = {
        "assessment": {
            "well_calibrated": False,
            "mean_coverage_deviation": 0.12,
            "calibration_diagnosis": "under-confident (predictions too uncertain)",
        }
    }
    report = check_diagnostics(diagnostics=_healthy_diagnostics(), calibration=cal)
    assert report["calibration"]["rating"] == "fair"
    assert report["calibration"]["rating"] != "poor"
    blob = " ".join(suggest_next_steps(report))
    assert "tighten prior" not in blob.lower()
    assert "not a prior-tuning signal" in blob
    assert "SBC" in blob
    assert "observation model" in blob.lower()


def test_positive_coverage_error_without_label_is_not_poor() -> None:
    cal = {
        "assessment": {
            "well_calibrated": False,
            "mean_coverage_deviation": 0.2,
            "calibration_diagnosis": "",
        }
    }
    report = check_diagnostics(diagnostics=_healthy_diagnostics(), calibration=cal)
    assert report["calibration"]["rating"] != "poor"
    blob = " ".join(suggest_next_steps(report)).lower()
    assert "tighten prior" not in blob
    assert "not a prior-tuning signal" in blob


def test_overconfident_stays_poor() -> None:
    cal = {
        "assessment": {
            "well_calibrated": False,
            "mean_coverage_deviation": -0.12,
            "calibration_diagnosis": "over-confident (predictions too certain)",
        }
    }
    report = check_diagnostics(diagnostics=_healthy_diagnostics(), calibration=cal)
    assert report["calibration"]["rating"] == "poor"
    blob = " ".join(suggest_next_steps(report)).lower()
    assert "tighten prior" not in blob
    assert "over-confident" in blob
