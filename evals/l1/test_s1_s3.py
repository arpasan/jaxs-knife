"""L1: S1–S3 script gates (ratings, tail ESS, chain floor).

JSON cases do not need ArviZ. InferenceData cases skip when ArviZ is absent.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from check_diagnostics import check_diagnostics, suggest_next_steps


# ==================================================
# JSON fixtures
# ==================================================


def _passing_diagnostics() -> Dict[str, Any]:
    return {
        "convergence": {"all_ok": True},
        "loo": {
            "computed": True,
            "pareto_k": {
                "max": 0.2,
                "n_bad": 0,
                "n_marginal": 0,
                "n_nonfinite": 0,
                "ok": True,
            },
        },
        "posterior_predictive": {"available": True},
    }


def _underconfident_calibration() -> Dict[str, Any]:
    """In-sample over-coverage. calibration_check labels this under-confident."""
    return {
        "assessment": {
            "well_calibrated": False,
            "mean_coverage_deviation": 0.12,
            "calibration_diagnosis": "under-confident (predictions too uncertain)",
        }
    }


# ==================================================
# S1a — empty or missing convergence
# ==================================================


@pytest.mark.parametrize(
    "diagnostics",
    [
        {},
        {"convergence": {}},
        {"convergence": None},
        {"loo": {"computed": False}, "posterior_predictive": {"available": True}},
    ],
)
def test_empty_or_missing_conv_is_not_excellent(diagnostics: Dict[str, Any]) -> None:
    report = check_diagnostics(diagnostics=diagnostics)
    assert report["convergence"]["rating"] == "unknown"
    assert report["convergence"]["rating"] != "excellent"
    summary = report["summary"]["convergence"]
    assert "unknown" in summary.lower()
    assert "All convergence diagnostics passed" not in summary
    steps = suggest_next_steps(report)
    assert not any("acceptable bounds" in step for step in steps)


def test_all_ok_still_rates_excellent() -> None:
    report = check_diagnostics(diagnostics=_passing_diagnostics())
    assert report["convergence"]["rating"] == "excellent"
    steps = suggest_next_steps(report)
    assert any("acceptable bounds" in step for step in steps)


# ==================================================
# S2 — over-coverage is not a prior-tightening signal
# ==================================================


def test_overcoverage_advice_is_not_tighten_priors() -> None:
    report = check_diagnostics(
        diagnostics=_passing_diagnostics(),
        calibration=_underconfident_calibration(),
    )
    steps = suggest_next_steps(report)
    blob = "\n".join(steps)
    assert "tighten priors" not in blob.lower()
    assert (
        "Single-dataset over-coverage; not a prior-tuning signal. "
        "Check observation model / SBC for coverage."
    ) in blob


def test_overconfident_advice_stays_on_the_likelihood() -> None:
    calibration = {
        "assessment": {
            "well_calibrated": False,
            "mean_coverage_deviation": -0.12,
            "calibration_diagnosis": "over-confident (predictions too certain)",
        }
    }
    report = check_diagnostics(
        diagnostics=_passing_diagnostics(),
        calibration=calibration,
    )
    steps = suggest_next_steps(report)
    blob = "\n".join(steps)
    assert "likelihood too narrow" in blob
    assert "not a prior-tuning signal" not in blob
    assert "tighten priors" not in blob.lower()


# ==================================================
# S1b / S3 — InferenceData
# ==================================================


def _arviz_stack() -> Any:
    pytest.importorskip("arviz")
    pytest.importorskip("numpy")
    import diagnose_model as dm
    from to_inference_data import from_blackjax

    return dm, from_blackjax


def _idata(n_chains: int, n_draws: int = 800) -> Any:
    import numpy as np

    _dm, from_blackjax = _arviz_stack()
    del _dm
    rng = np.random.default_rng(1)
    return from_blackjax(
        {
            "mu": rng.normal(size=(n_chains, n_draws)),
            "sigma": np.exp(rng.normal(size=(n_chains, n_draws)) * 0.05),
        },
        sample_stats={"diverging": np.zeros((n_chains, n_draws), dtype=bool)},
    )


def test_ess_tail_is_not_an_alias_of_bulk() -> None:
    dm, _from_blackjax = _arviz_stack()
    report = dm.check_convergence(_idata(4))
    bulk = report["ess_bulk"]["problematic_params"]
    tail = report["ess_tail"]["problematic_params"]
    assert bulk is not tail


def test_ess_tail_computed_apart_on_diagnose_path(monkeypatch: pytest.MonkeyPatch) -> None:
    dm, _from_blackjax = _arviz_stack()
    monkeypatch.setattr(dm, "HAS_DIAGNOSE", True)

    def _diagnose(data: object, **kwargs: object) -> tuple:
        return False, {
            "rhat": {"bad_params": [], "rhat_values": None},
            "ess": {"bad_params": ["copied_bulk"]},
            "treedepth": {"n_max": 0, "pct": 0.0},
            "bfmi": {"failed_chains": []},
        }

    def _ess(idata: object, method: str = "bulk", **kwargs: object) -> Any:
        import xarray as xr

        if method == "tail":
            return xr.Dataset({"mu": 9000.0, "sigma": 10.0})
        if method == "bulk":
            return xr.Dataset({"mu": 10.0, "sigma": 9000.0})
        raise AssertionError(method)

    monkeypatch.setattr(dm.azs, "diagnose", _diagnose)
    monkeypatch.setattr(dm.az, "ess", _ess)
    report = dm.check_convergence(_idata(4, n_draws=30))
    assert report["ess_bulk"]["problematic_params"] == ["mu"]
    assert report["ess_tail"]["problematic_params"] == ["sigma"]
    assert "copied_bulk" not in report["ess_bulk"]["problematic_params"]
    assert "copied_bulk" not in report["ess_tail"]["problematic_params"]
    assert report["ess_bulk"]["ok"] is False
    assert report["ess_tail"]["ok"] is False
    assert report["chain_floor"]["ok"] is True


def test_ess_tail_computed_apart_on_summary_path(monkeypatch: pytest.MonkeyPatch) -> None:
    dm, _from_blackjax = _arviz_stack()
    monkeypatch.setattr(dm, "HAS_DIAGNOSE", False)

    def _summary(idata: object, **kwargs: object) -> Any:
        import pandas as pd

        return pd.DataFrame(
            {
                "r_hat": [1.0, 1.0],
                "ess_bulk": [10.0, 9000.0],
                "ess_tail": [9000.0, 10.0],
            },
            index=["mu", "sigma"],
        )

    monkeypatch.setattr(dm.az, "summary", _summary)
    report = dm.check_convergence(_idata(4, n_draws=30))
    assert list(report["ess_bulk"]["problematic_params"]) == ["mu"]
    assert list(report["ess_tail"]["problematic_params"]) == ["sigma"]
    assert report["ess_bulk"]["problematic_params"] is not report["ess_tail"]["problematic_params"]


@pytest.mark.parametrize("n_chains", [1, 2, 3])
def test_nuts_ess_gate_fails_below_four_chains(n_chains: int) -> None:
    dm, _from_blackjax = _arviz_stack()
    report = dm.check_convergence(_idata(n_chains))
    assert report["sampler_method"] == "nuts"
    assert report["n_chains"] == n_chains
    assert report["chain_floor"]["ok"] is False
    assert report["chain_floor"]["exempt"] is False
    assert report["ess_bulk"]["ok"] is False
    assert report["ess_tail"]["ok"] is False
    assert report["all_ok"] is False
    assert report["ess_bulk"]["problematic_params"] == []
    assert report["ess_tail"]["problematic_params"] == []
    rated = check_diagnostics(
        diagnostics={
            "convergence": report,
            "loo": {"computed": False},
            "posterior_predictive": {"available": True},
        }
    )
    assert rated["convergence"]["rating"] != "excellent"
    steps = suggest_next_steps(rated)
    blob = "\n".join(steps)
    assert "below 4" in blob
    assert "acceptable bounds" not in blob
    assert "more draws" not in blob.lower()


def test_four_chains_still_clear_the_ess_gate() -> None:
    dm, _from_blackjax = _arviz_stack()
    report = dm.check_convergence(_idata(4))
    assert report["chain_floor"]["ok"] is True
    assert report["ess_bulk"]["ok"] is True
    assert report["ess_tail"]["ok"] is True
    assert report["all_ok"] is True


@pytest.mark.parametrize("method", ["pathfinder", "vi", "laplace", "ADVI", "SVI", "Pathfinder"])
@pytest.mark.parametrize("n_chains", [1, 3])
def test_pathfinder_vi_laplace_exempt_from_chain_floor(method: str, n_chains: int) -> None:
    dm, _from_blackjax = _arviz_stack()
    report = dm.check_convergence(_idata(n_chains), method=method)
    assert report["chain_floor"]["exempt"] is True
    assert report["chain_floor"]["ok"] is True
    assert report["ess_bulk"]["ok"] is True
    assert report["ess_tail"]["ok"] is True
    assert report["all_ok"] is True
    rated = check_diagnostics(
        diagnostics={
            "convergence": report,
            "posterior_predictive": {"available": True},
        }
    )
    steps = suggest_next_steps(rated)
    assert not any("below 4" in step for step in steps)


def test_method_label_on_idata_exempts_laplace() -> None:
    dm, _from_blackjax = _arviz_stack()
    idata = _idata(1)
    idata.posterior.attrs["inference_method"] = "laplace"
    report = dm.check_convergence(idata)
    assert report["sampler_method"] == "laplace"
    assert report["chain_floor"]["exempt"] is True
    assert report["ess_bulk"]["ok"] is True
    assert report["all_ok"] is True


def test_explicit_method_overrides_idata_label() -> None:
    dm, _from_blackjax = _arviz_stack()
    idata = _idata(1)
    idata.posterior.attrs["inference_method"] = "laplace"
    report = dm.check_convergence(idata, method="nuts")
    assert report["sampler_method"] == "nuts"
    assert report["chain_floor"]["ok"] is False
    assert report["ess_bulk"]["ok"] is False
    assert report["all_ok"] is False


def test_generate_report_forwards_method() -> None:
    dm, _from_blackjax = _arviz_stack()
    report = dm.generate_report(_idata(1), method="vi")
    floor = report["convergence"]["chain_floor"]
    assert floor["exempt"] is True
    assert floor["ok"] is True
    assert report["convergence"]["all_ok"] is True


def test_summary_path_chain_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    dm, _from_blackjax = _arviz_stack()
    monkeypatch.setattr(dm, "HAS_DIAGNOSE", False)
    nuts = dm.check_convergence(_idata(2, n_draws=400))
    assert nuts["chain_floor"]["ok"] is False
    assert nuts["ess_bulk"]["ok"] is False
    assert nuts["all_ok"] is False
    approx = dm.check_convergence(_idata(2, n_draws=400), method="laplace")
    assert approx["chain_floor"]["exempt"] is True
    assert approx["ess_bulk"]["ok"] is True
    assert approx["ess_tail"]["ok"] is True
    assert approx["all_ok"] is True


def test_split_ess_failure_does_not_copy_bulk(monkeypatch: pytest.MonkeyPatch) -> None:
    dm, _from_blackjax = _arviz_stack()
    monkeypatch.setattr(dm, "HAS_DIAGNOSE", True)

    def _diagnose(data: object, **kwargs: object) -> tuple:
        return False, {
            "rhat": {"bad_params": [], "rhat_values": None},
            "ess": {"bad_params": ["copied_bulk"]},
            "treedepth": {"n_max": 0, "pct": 0.0},
            "bfmi": {"failed_chains": []},
        }

    def _boom(idata: object, **kwargs: object) -> Any:
        raise RuntimeError("ess unavailable")

    monkeypatch.setattr(dm.azs, "diagnose", _diagnose)
    monkeypatch.setattr(dm.az, "ess", _boom)
    report = dm.check_convergence(_idata(4, n_draws=20))
    assert report["ess_bulk"]["problematic_params"] == ["copied_bulk"]
    assert report["ess_tail"]["problematic_params"] == ["ess_tail_not_computed"]
    assert report["ess_tail"]["ok"] is False
    assert report["ess_tail"]["problematic_params"] != report["ess_bulk"]["problematic_params"]
