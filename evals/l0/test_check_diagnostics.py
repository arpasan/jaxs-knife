"""L0: qualitative ratings from JSON only (no ArviZ, no live MCMC)."""

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
            "pareto_k": {
                "max": 0.3,
                "n_bad": 0,
                "n_marginal": 0,
                "n_nonfinite": 0,
                "ok": True,
            },
        },
        "posterior_predictive": {"available": True, "variables": ["y_rep"]},
        "overall": {"ok": True, "issues": [], "recommendation": "ok"},
    }


def _divergent_diagnostics() -> dict:
    healthy = _healthy_diagnostics()
    healthy["convergence"] = {
        "all_ok": False,
        "method": "manual",
        "rhat": {"ok": True, "max": 1.002, "problematic_params": []},
        "ess_bulk": {"ok": True, "min": 800, "problematic_params": []},
        "ess_tail": {"ok": True, "min": 700, "problematic_params": []},
        "divergences": {"count": 40, "pct": 2.0, "ok": False},
    }
    healthy["overall"] = {"ok": False, "issues": ["divergences"], "recommendation": "stop"}
    return healthy


# ==================================================
# Tests
# ==================================================


def test_healthy_rates_excellent() -> None:
    report = check_diagnostics(diagnostics=_healthy_diagnostics())
    assert report["convergence"]["rating"] == "excellent"
    assert report["loo"]["rating"] == "excellent"
    steps = suggest_next_steps(report)
    assert steps
    assert "acceptable bounds" in steps[0]


def test_divergences_rate_poor_and_refuse() -> None:
    report = check_diagnostics(diagnostics=_divergent_diagnostics())
    assert report["convergence"]["rating"] == "poor"
    steps = suggest_next_steps(report)
    assert any("Do not interpret" in s or "reparameterize" in s.lower() for s in steps)


def test_high_rhat_is_poor() -> None:
    diag = _healthy_diagnostics()
    diag["convergence"] = {
        "all_ok": False,
        "rhat": {"ok": False, "max": 1.12, "problematic_params": ["theta"]},
        "ess_bulk": {"ok": True, "problematic_params": []},
        "ess_tail": {"ok": True, "problematic_params": []},
        "divergences": {"count": 0, "pct": 0.0, "ok": True},
    }
    report = check_diagnostics(diagnostics=diag)
    assert report["convergence"]["rating"] == "poor"


def test_same_json_same_ratings() -> None:
    """CmdStan vs nutpie traces of the same draws must share a rating class."""
    diag = _healthy_diagnostics()
    a = check_diagnostics(diagnostics=diag)
    b = check_diagnostics(diagnostics=diag)
    assert a["convergence"]["rating"] == b["convergence"]["rating"]
    assert a["loo"]["rating"] == b["loo"]["rating"]


def test_calibration_overconfident() -> None:
    cal = {
        "assessment": {
            "well_calibrated": False,
            "mean_coverage_deviation": -0.08,
            "calibration_diagnosis": "over-confident (predictions too certain)",
        }
    }
    report = check_diagnostics(diagnostics=_healthy_diagnostics(), calibration=cal)
    assert report["calibration"]["rating"] == "poor"
    steps = suggest_next_steps(report)
    assert any("over-confident" in s for s in steps)
