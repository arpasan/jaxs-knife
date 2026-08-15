"""Live NUTS: CmdStanPy and BlackJAX must both sample. No skips."""

from __future__ import annotations

import sys
from pathlib import Path

from live_nuts import ARTIFACTS, fit_blackjax, fit_stan, simulate_data, write_run

L2 = Path(__file__).resolve().parents[1] / "l2"
sys.path.insert(0, str(L2))

from band_b import assess_recovery, posterior_from_idata

TRUTH = {"mu": 0.3, "sigma": 1.0}


def _assert_healthy(report: dict, engine: str) -> None:
    conv = report["convergence"]["rating"]
    assert conv in {"excellent", "good"}, f"{engine} convergence={conv}"
    n_div = report["diagnostics"]["convergence"]["divergences"]["count"]
    assert n_div == 0, f"{engine} divergences={n_div}"
    assert report["posterior_predictive"]["available"] is True


def _assert_recovers(idata: object, engine: str) -> None:
    posterior = posterior_from_idata(idata, ("mu", "sigma"))
    rec = assess_recovery(posterior, TRUTH, nominal=0.94)
    assert rec["passed"] is True, f"{engine} Band B {rec}"


def test_live_cmdstan_nuts() -> None:
    y = simulate_data()
    idata, report = fit_stan(y)
    write_run("stan", idata, report)
    _assert_healthy(report, "stan")
    _assert_recovers(idata, "stan")
    assert (ARTIFACTS / "stan" / "inference_data.nc").exists()


def test_live_blackjax_nuts() -> None:
    y = simulate_data()
    idata, report = fit_blackjax(y)
    write_run("jax", idata, report)
    _assert_healthy(report, "jax")
    _assert_recovers(idata, "jax")
    assert (ARTIFACTS / "jax" / "inference_data.nc").exists()
