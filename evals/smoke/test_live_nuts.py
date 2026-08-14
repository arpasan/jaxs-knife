"""Live NUTS: CmdStanPy and BlackJAX must both sample. No skips."""

from __future__ import annotations

from pathlib import Path

from live_nuts import ARTIFACTS, fit_blackjax, fit_stan, simulate_data, write_run


def _assert_healthy(report: dict, engine: str) -> None:
    conv = report["convergence"]["rating"]
    assert conv in {"excellent", "good"}, f"{engine} convergence={conv}"
    n_div = report["diagnostics"]["convergence"]["divergences"]["count"]
    assert n_div == 0, f"{engine} divergences={n_div}"
    assert report["posterior_predictive"]["available"] is True


def test_live_cmdstan_nuts() -> None:
    y = simulate_data()
    idata, report = fit_stan(y)
    write_run("stan", idata, report)
    _assert_healthy(report, "stan")
    assert (ARTIFACTS / "stan" / "inference_data.nc").exists()


def test_live_blackjax_nuts() -> None:
    y = simulate_data()
    idata, report = fit_blackjax(y)
    write_run("jax", idata, report)
    _assert_healthy(report, "jax")
    assert (ARTIFACTS / "jax" / "inference_data.nc").exists()
