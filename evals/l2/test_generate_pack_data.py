"""Fixture screens for generated and coverage-checked pack CSVs."""

from __future__ import annotations

import numpy as np

from generate_pack_data import (
    _eiv_latent_moments,
    _read_eiv,
    check_a1,
    check_e1,
    check_h1,
    check_j1,
    check_k1,
    check_m1,
    PACKS,
)


def test_e1_naive_ols_misses_eiv_covers() -> None:
    assert check_e1() is True


def test_e1_reference_estimates_latent_moments() -> None:
    x_obs = np.asarray([2.0, 2.0, 2.0, 4.0], dtype=float)
    x_se = np.asarray([0.1, 0.1, 0.1, 0.1], dtype=float)
    mu_x, var_x = _eiv_latent_moments(x_obs, x_se)
    assert abs(mu_x - 2.5) < 1e-12
    var_obs = float(x_obs.var(ddof=1))
    assert abs(var_x - max(var_obs - 0.01, 1e-4)) < 1e-12
    x, se, _y = _read_eiv(PACKS / "E1" / "data.csv")
    mu_hat, var_hat = _eiv_latent_moments(x, se)
    pinned = abs(mu_hat) < 1e-9 and abs(var_hat - 1.0) < 1e-9
    assert not pinned, "E1 screen must not pin generating mu_x=0, var_x=1"


def test_h1_complete_pool_misses_hier_covers() -> None:
    assert check_h1() is True


def test_a1_naive_rate_misses_assay_covers() -> None:
    assert check_a1() is True


def test_k1_single_normal_misses_em_covers() -> None:
    assert check_k1() is True


def test_j1_gaussian_q95_misses_lognormal_covers() -> None:
    assert check_j1() is True


def test_m1_complete_case_misses_tobit_covers() -> None:
    assert check_m1() is True
