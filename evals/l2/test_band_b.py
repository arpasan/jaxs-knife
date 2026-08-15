"""Band B recovery on constructed draws."""

from __future__ import annotations

import numpy as np

from band_b import assess_recovery, hdi


def test_hdi_contains_center() -> None:
    rng = np.random.default_rng(0)
    draws = rng.normal(0.3, 0.1, size=(4, 400))
    lo, hi = hdi(draws, 0.94)
    assert lo <= 0.3 <= hi


def test_recovery_passes_when_truth_inside() -> None:
    rng = np.random.default_rng(1)
    posterior = {
        "mu": rng.normal(0.3, 0.08, size=(4, 300)),
        "sigma": rng.normal(1.0, 0.08, size=(4, 300)),
    }
    report = assess_recovery(posterior, {"mu": 0.3, "sigma": 1.0}, nominal=0.94)
    assert report["passed"] is True


def test_recovery_fails_when_truth_outside() -> None:
    rng = np.random.default_rng(2)
    posterior = {"mu": rng.normal(2.0, 0.05, size=(4, 200))}
    report = assess_recovery(posterior, {"mu": 0.0}, nominal=0.94)
    assert report["passed"] is False


def test_recovery_fails_when_parameter_missing() -> None:
    report = assess_recovery({"mu": np.zeros(10)}, {"sigma": 1.0}, nominal=0.94)
    assert report["passed"] is False
    assert report["parameters"]["sigma"]["ok"] is False
