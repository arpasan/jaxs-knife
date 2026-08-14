"""L0: nominal HDI coverage on synthetic PPC."""

from __future__ import annotations

import numpy as np
import pytest

az = pytest.importorskip("arviz")

from calibration_check import generate_calibration_report
from to_inference_data import from_blackjax


def test_well_calibrated_coverage() -> None:
    rng = np.random.default_rng(0)
    n_chains, n_draws, n_obs = 4, 300, 40
    y = rng.normal(0.0, 1.0, size=(n_obs,))
    y_rep = rng.normal(0.0, 1.0, size=(n_chains, n_draws, n_obs))
    mu = rng.normal(0.0, 0.1, size=(n_chains, n_draws))
    idata = from_blackjax(
        {"mu": mu},
        observed_data={"y": y},
        posterior_predictive={"y": y_rep},
    )
    report = generate_calibration_report(idata, var_name="y", nominal=0.94)
    assert report["assessment"]["calibration_diagnosis"] in {
        "well-calibrated",
        "under-confident (predictions too uncertain)",
        "over-confident (predictions too certain)",
    }
    assert "mean_coverage_deviation" in report["assessment"]
