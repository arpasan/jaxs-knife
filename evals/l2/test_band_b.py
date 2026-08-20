"""Band B recovery on constructed draws."""

from __future__ import annotations

import numpy as np

from band_b import assess_recovery, hdi
from generate_pack_data import J1_Q95


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


class _Var:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _Post:
    def __init__(self, mapping: dict) -> None:
        self.data_vars = list(mapping)
        self._m = {k: _Var(v) for k, v in mapping.items()}

    def __getitem__(self, name: str) -> _Var:
        return self._m[name]


class _Idata:
    def __init__(self, **vars: np.ndarray) -> None:
        self.posterior = _Post(vars)


def test_posterior_from_idata_falls_back_to_getitem() -> None:
    from band_b import posterior_from_idata

    class _Tree:
        def __getitem__(self, name: str) -> _Post:
            if name != "posterior":
                raise KeyError(name)
            return _Post({"mu": np.ones((2, 8))})

    out = posterior_from_idata(_Tree(), ("mu",))
    assert "mu" in out
    assert out["mu"].shape == (2, 8)


def test_posterior_from_idata_omits_missing_names() -> None:
    from band_b import posterior_from_idata

    idata = _Idata(mu=np.zeros((2, 10)))
    out = posterior_from_idata(idata, ("mu", "sigma"))
    assert "mu" in out
    assert "sigma" not in out


def test_posterior_from_idata_aliases_ld50() -> None:
    from band_b import posterior_from_idata

    draws = np.full((2, 20), 0.25)
    idata = _Idata(LD50=draws)
    out = posterior_from_idata(idata, ("ld50",))
    assert "ld50" in out
    assert float(out["ld50"].mean()) == 0.25


def test_posterior_from_idata_reads_generated_q95() -> None:
    from band_b import posterior_from_idata

    class _IdataPPC:
        def __init__(self) -> None:
            self.posterior = _Post({"mu": np.zeros((2, 8))})
            self.posterior_predictive = _Post(
                {"q95": np.linspace(8.4, 10.0, 16).reshape(2, 8)}
            )

    out = posterior_from_idata(_IdataPPC(), ("q95",))
    assert out["q95"].size == 16
    assert 8.4 <= float(out["q95"].mean()) <= 10.0


def test_scalar_q95_is_rejected() -> None:
    report = assess_recovery({"q95": np.array([9.21])}, {"q95": 9.21}, nominal=0.94)
    assert report["passed"] is False
    assert "plug-in" in report["parameters"]["q95"]["error"]


def test_q95_from_y_rep_draws() -> None:
    from band_b import posterior_from_idata

    rng = np.random.default_rng(4)
    y_rep = rng.lognormal(mean=0.0, sigma=1.35, size=(4, 80, 200))
    idata = _IdataPPCBoth(q95=np.array([9.21]), y_rep=y_rep)
    out = posterior_from_idata(idata, ("q95",))
    assert out["q95"].size >= 32
    report = assess_recovery(out, {"q95": J1_Q95}, nominal=0.94)
    assert report["passed"] is True


def test_tiled_scalar_q95_is_rejected() -> None:
    report = assess_recovery(
        {"q95": np.full((4, 200), 9.21)},
        {"q95": 9.21},
        nominal=0.94,
    )
    assert report["passed"] is False
    assert "plug-in" in report["parameters"]["q95"]["error"]


class _IdataPPCBoth:
    def __init__(self, q95: np.ndarray, y_rep: np.ndarray) -> None:
        self.posterior = _Post({"q95": q95})
        self.posterior_predictive = _Post({"y_rep": y_rep})


def test_posterior_from_idata_splits_ordered_mu() -> None:
    from band_b import posterior_from_idata

    mu = np.stack(
        [np.full((2, 15), -1.1), np.full((2, 15), 1.4)],
        axis=-1,
    )
    idata = _Idata(mu=mu)
    out = posterior_from_idata(idata, ("mu1", "mu2"))
    assert abs(float(out["mu1"].mean()) + 1.1) < 1e-12
    assert abs(float(out["mu2"].mean()) - 1.4) < 1e-12
