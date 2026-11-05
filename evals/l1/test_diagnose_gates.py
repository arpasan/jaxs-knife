"""L1: tail ESS is not a copy of bulk, and NUTS has a 4-chain floor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

az = pytest.importorskip("arviz")

from diagnose_model import generate_report
from to_inference_data import from_blackjax


# ==================================================
# Fixtures
# ==================================================


def _four_chain_idata(seed: int = 0) -> object:
    rng = np.random.default_rng(seed)
    n_chains, n_draws = 4, 250
    diverging = np.zeros((n_chains, n_draws), dtype=bool)
    return from_blackjax(
        {
            "mu": rng.normal(0.0, 0.15, size=(n_chains, n_draws)),
            "sigma": np.exp(rng.normal(0.0, 0.05, size=(n_chains, n_draws))),
        },
        sample_stats={"diverging": diverging},
    )


def _one_chain_bimodal(seed: int = 0) -> object:
    """One chain that visits both modes. Split-R-hat can still look fine.

    Numeric ESS clears 100 draws. The NUTS chain floor is what must fail.
    """
    rng = np.random.default_rng(seed)
    n_draws = 2000
    component = rng.integers(0, 2, size=(1, n_draws))
    mu = np.where(
        component == 0,
        rng.normal(-2.0, 0.05, size=(1, n_draws)),
        rng.normal(2.0, 0.05, size=(1, n_draws)),
    )
    sigma = np.exp(rng.normal(0.0, 0.02, size=(1, n_draws)))
    diverging = np.zeros((1, n_draws), dtype=bool)
    return from_blackjax(
        {"mu": mu, "sigma": sigma},
        sample_stats={"diverging": diverging},
    )


def _quiet_diag(**extra: object) -> dict:
    diag = {
        "rhat": {"bad_params": [], "rhat_values": None},
        "ess": {"bad_params": []},
        "treedepth": {"n_max": 0, "pct": 0.0},
        "bfmi": {"failed_chains": []},
    }
    diag.update(extra)
    return diag


# ==================================================
# S1b — ess_tail is not a copy of bulk
# ==================================================


def test_ess_tail_diag_is_not_copied_from_bulk(monkeypatch: pytest.MonkeyPatch) -> None:
    import diagnose_model as dm

    monkeypatch.setattr(dm, "HAS_DIAGNOSE", True)

    def fake_diagnose(*args: object, **kwargs: object) -> tuple:
        return True, _quiet_diag(
            ess={"bad_params": ["mu"]},
            ess_tail={"bad_params": ["sigma"]},
        )

    def ess_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("ess_tail block should be used as given")

    monkeypatch.setattr(dm, "azs", SimpleNamespace(diagnose=fake_diagnose))
    monkeypatch.setattr(dm.az, "ess", ess_must_not_run)
    report = dm.check_convergence(_four_chain_idata())
    assert report["ess_bulk"]["problematic_params"] == ["mu"]
    assert report["ess_tail"]["problematic_params"] == ["sigma"]
    assert report["ess_bulk"]["ok"] is False
    assert report["ess_tail"]["ok"] is False
    assert "n_chains<4" not in report["ess_tail"]["problematic_params"]


def test_ess_tail_computed_when_diag_has_no_tail_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xarray as xr

    import diagnose_model as dm

    monkeypatch.setattr(dm, "HAS_DIAGNOSE", True)

    def fake_diagnose(*args: object, **kwargs: object) -> tuple:
        return False, _quiet_diag(ess={"bad_params": ["mu"]})

    def fake_ess(*args: object, **kwargs: object) -> xr.Dataset:
        assert kwargs.get("method") == "tail"
        return xr.Dataset({"mu": 5000.0, "sigma": 5.0})

    monkeypatch.setattr(dm, "azs", SimpleNamespace(diagnose=fake_diagnose))
    monkeypatch.setattr(dm.az, "ess", fake_ess)
    report = dm.check_convergence(_four_chain_idata())
    assert report["ess_bulk"]["problematic_params"] == ["mu"]
    assert report["ess_tail"]["problematic_params"] == ["sigma"]
    assert report["ess_bulk"]["problematic_params"] != report["ess_tail"]["problematic_params"]


# ==================================================
# S3 — NUTS chain floor; Pathfinder / VI / Laplace exempt
# ==================================================


def test_one_chain_bimodal_nuts_fails_chain_floor() -> None:
    report = generate_report(_one_chain_bimodal())
    conv = report["convergence"]
    assert conv["n_chains"] == 1
    assert conv["inference_method"] == "nuts"
    assert conv["ess_bulk"]["ok"] is False
    assert conv["ess_tail"]["ok"] is False
    assert "n_chains<4" in conv["ess_bulk"]["problematic_params"]
    assert "n_chains<4" in conv["ess_tail"]["problematic_params"]
    assert conv["all_ok"] is False


@pytest.mark.parametrize("method", ["pathfinder", "vi", "laplace", "advi"])
def test_approx_one_chain_skips_chain_floor(method: str) -> None:
    report = generate_report(_one_chain_bimodal(), inference_method=method)
    conv = report["convergence"]
    assert conv["inference_method"] in {"pathfinder", "vi", "laplace"}
    assert "n_chains<4" not in conv["ess_bulk"]["problematic_params"]
    assert "n_chains<4" not in conv["ess_tail"]["problematic_params"]
    assert conv["ess_bulk"]["ok"] is True
    assert conv["ess_tail"]["ok"] is True


def test_method_label_read_from_posterior_attrs() -> None:
    idata = _one_chain_bimodal()
    idata.posterior.attrs["inference_method"] = "Laplace"
    report = generate_report(idata)
    conv = report["convergence"]
    assert conv["inference_method"] == "laplace"
    assert conv["ess_bulk"]["ok"] is True
    assert "n_chains<4" not in conv["ess_bulk"]["problematic_params"]


def test_explicit_method_overrides_attrs() -> None:
    idata = _one_chain_bimodal()
    idata.attrs["inference_method"] = "pathfinder"
    report = generate_report(idata, inference_method="nuts")
    conv = report["convergence"]
    assert conv["inference_method"] == "nuts"
    assert conv["ess_bulk"]["ok"] is False
    assert "n_chains<4" in conv["ess_bulk"]["problematic_params"]


def test_summary_fallback_one_chain_nuts_fails_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    import diagnose_model as dm

    monkeypatch.setattr(dm, "HAS_DIAGNOSE", False)
    report = dm.generate_report(_one_chain_bimodal())
    conv = report["convergence"]
    assert conv["method"] == "manual"
    assert conv["ess_bulk"]["ok"] is False
    assert conv["ess_tail"]["ok"] is False
    assert "n_chains<4" in conv["ess_bulk"]["problematic_params"]
    assert "n_chains<4" in conv["ess_tail"]["problematic_params"]


def test_summary_fallback_pathfinder_skips_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    import diagnose_model as dm

    monkeypatch.setattr(dm, "HAS_DIAGNOSE", False)
    report = dm.generate_report(_one_chain_bimodal(), inference_method="pathfinder")
    conv = report["convergence"]
    assert conv["method"] == "manual"
    assert conv["inference_method"] == "pathfinder"
    assert conv["ess_bulk"]["ok"] is True
    assert conv["ess_tail"]["ok"] is True
    assert "n_chains<4" not in conv["ess_bulk"]["problematic_params"]
    assert "n_chains<4" not in conv["ess_tail"]["problematic_params"]
