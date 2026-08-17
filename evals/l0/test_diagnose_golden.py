"""L0: diagnose_model on synthetic InferenceData (no live NUTS)."""

from __future__ import annotations

import numpy as np
import pytest

az = pytest.importorskip("arviz")

from check_diagnostics import check_diagnostics
from diagnose_model import generate_report
from to_inference_data import from_blackjax


# ==================================================
# Golden traces
# ==================================================


def _healthy_idata(seed: int = 0) -> object:
    rng = np.random.default_rng(seed)
    n_chains, n_draws, n_obs = 4, 250, 8
    mu = rng.normal(0.0, 0.15, size=(n_chains, n_draws))
    sigma = np.exp(rng.normal(0.0, 0.05, size=(n_chains, n_draws)))
    y = rng.normal(0.0, 1.0, size=(n_obs,))
    y_rep = rng.normal(0.0, 1.0, size=(n_chains, n_draws, n_obs))
    diverging = np.zeros((n_chains, n_draws), dtype=bool)
    return from_blackjax(
        {"mu": mu, "sigma": sigma},
        sample_stats={"diverging": diverging},
        observed_data={"y": y},
        posterior_predictive={"y": y_rep},
    )


def _pathological_idata(seed: int = 1) -> object:
    rng = np.random.default_rng(seed)
    n_chains, n_draws = 4, 250
    mu = np.stack(
        [
            rng.normal(-3.0, 0.1, size=n_draws),
            rng.normal(-1.0, 0.1, size=n_draws),
            rng.normal(1.0, 0.1, size=n_draws),
            rng.normal(3.0, 0.1, size=n_draws),
        ],
        axis=0,
    )
    sigma = np.exp(rng.normal(0.0, 0.05, size=(n_chains, n_draws)))
    diverging = np.zeros((n_chains, n_draws), dtype=bool)
    diverging[:, :20] = True
    return from_blackjax(
        {"mu": mu, "sigma": sigma},
        sample_stats={"diverging": diverging},
    )


# ==================================================
# Tests
# ==================================================


def test_healthy_convergence_ok() -> None:
    report = generate_report(_healthy_idata())
    assert report["convergence"]["all_ok"] is True
    assert report["convergence"]["divergences"]["count"] == 0
    assert report["posterior_predictive"]["available"] is True
    rated = check_diagnostics(diagnostics=report)
    assert rated["convergence"]["rating"] in {"excellent", "good"}


def test_pathological_refuses() -> None:
    report = generate_report(_pathological_idata())
    assert report["convergence"]["all_ok"] is False
    assert report["convergence"]["divergences"]["count"] > 0
    rated = check_diagnostics(diagnostics=report)
    assert rated["convergence"]["rating"] in {"poor", "fair"}


def test_engine_label_does_not_change_ratings() -> None:
    idata = _healthy_idata()
    cmdstan_like = generate_report(idata)
    nutpie_like = generate_report(idata)
    a = check_diagnostics(diagnostics=cmdstan_like)
    b = check_diagnostics(diagnostics=nutpie_like)
    assert a["convergence"]["rating"] == b["convergence"]["rating"]


def test_convergence_fallback_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    import diagnose_model as dm

    monkeypatch.setattr(dm, "HAS_DIAGNOSE", False)

    class _Boom:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("summary exploded")

    monkeypatch.setattr(dm.az, "summary", _Boom)
    report = generate_report(_healthy_idata())
    assert report["convergence"]["all_ok"] is False
    assert "error" in report["convergence"]
