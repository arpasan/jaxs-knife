"""Regenerate pack CSVs. Hidden truth must be recoverable from the CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

_L2 = Path(__file__).resolve().parent
if str(_L2) not in sys.path:
    sys.path.insert(0, str(_L2))
from band_b import hdi

PACKS = Path(__file__).resolve().parent / "packs"
Array = NDArray[np.floating]
# Two-sided 94% Gaussian quantile (Φ^{-1}(0.97)).
Z_94 = 1.880793608151251

S4_ALPHA = 0.1
S4_BETA = 0.7
S4_SIGMA = 0.2
S8_MU = 0.15
S8_SIGMA = 0.45
S6_ALPHA = -0.4
S6_BETA = 1.5
S6_LD50 = -S6_ALPHA / S6_BETA
S7_MU1 = -1.1
S7_MU2 = 1.4
S7_WEIGHT = 0.4
S7_SIGMA = 0.5


def _z_nominal(nominal: float) -> float:
    """Two-sided Gaussian quantile for a ``nominal`` interval."""
    if abs(nominal - 0.94) > 1e-12:
        raise ValueError("this helper is pinned to a 94% interval")
    return Z_94


def _chi2_ppf(p: float, df: float) -> float:
    """Chi-square quantile. SciPy if present; Wilson–Hilferty otherwise."""
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(p, df))
    except Exception:
        from scipy.stats import norm

        z = float(norm.ppf(p))
        return float(df * (1.0 - 2.0 / (9.0 * df) + z * np.sqrt(2.0 / (9.0 * df))) ** 3)


def _write_xy(path: Path, x: Array, y: Array) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        for xi, yi in zip(x, y):
            writer.writerow([f"{float(xi):.6f}", f"{float(yi):.6f}"])


def _write_y(path: Path, y: Array) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["y"])
        for yi in y:
            writer.writerow([f"{float(yi):.6f}"])


def _read_xy(path: Path) -> Tuple[Array, Array]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    x = np.asarray([float(r["x"]) for r in rows], dtype=float)
    y = np.asarray([float(r["y"]) for r in rows], dtype=float)
    return x, y


def _read_y(path: Path) -> Array:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return np.asarray([float(r["y"]) for r in rows], dtype=float)


def _ols_covers(
    x: Array,
    y: Array,
    *,
    alpha: float,
    beta: float,
    nominal: float = 0.94,
) -> bool:
    """True when a normal-approx interval contains both slope truths."""
    design = np.column_stack([np.ones(len(x)), x])
    hat, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ hat
    n, p = design.shape
    cov = (resid.dot(resid) / (n - p)) * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(cov))
    z = _z_nominal(nominal)
    return bool(
        abs(float(hat[0]) - alpha) <= z * float(se[0])
        and abs(float(hat[1]) - beta) <= z * float(se[1])
    )


def _sigma_chi2_covers(
    resid: Array,
    df: int,
    truth: float,
    nominal: float = 0.94,
) -> bool:
    """Jeffreys / scaled-inv-χ² 94% interval for residual scale."""
    if df < 2:
        return False
    s2 = float(np.dot(resid, resid) / df)
    tail = (1.0 - nominal) / 2.0
    lo = float(np.sqrt(df * s2 / _chi2_ppf(1.0 - tail, float(df))))
    hi = float(np.sqrt(df * s2 / _chi2_ppf(tail, float(df))))
    return lo <= truth <= hi


def _s4_covers(x: Array, y: Array) -> bool:
    design = np.column_stack([np.ones(len(x)), x])
    hat, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ hat
    df = int(len(y) - design.shape[1])
    return _ols_covers(x, y, alpha=S4_ALPHA, beta=S4_BETA) and _sigma_chi2_covers(
        resid, df, S4_SIGMA
    )


def _mean_covers(y: Array, truth: float, nominal: float = 0.94) -> bool:
    se = float(y.std(ddof=1) / np.sqrt(len(y)))
    return bool(abs(float(y.mean()) - truth) <= _z_nominal(nominal) * se)


def _normal_exp_reference_covers(
    y: Array,
    truth: Mapping[str, float],
    *,
    nominal: float = 0.94,
) -> bool:
    """Grid posterior under skill defaults: μ ~ N(0, 2.5), σ ~ Exp(1)."""
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    mus = np.linspace(-1.5, 1.8, 201)
    sigmas = np.linspace(0.08, 1.6, 201)
    y2 = float(np.dot(y, y))
    ysum = float(y.sum())
    logp = np.empty((mus.size, sigmas.size), dtype=float)
    log_sig = np.log(sigmas)
    for i, mu in enumerate(mus):
        sse = y2 - 2.0 * mu * ysum + n * mu * mu
        logp[i] = (
            -n * log_sig
            - 0.5 * sse / (sigmas**2)
            - 0.5 * (mu / 2.5) ** 2
            - sigmas
        )
    logp -= float(logp.max())
    weight = np.exp(logp)
    weight /= float(weight.sum())
    rng = np.random.default_rng(0)
    flat = weight.ravel()
    idx = rng.choice(flat.size, size=6000, p=flat)
    mu_s = mus[idx // sigmas.size]
    sig_s = sigmas[idx % sigmas.size]
    for name, draws in (("mu", mu_s), ("sigma", sig_s)):
        lo, hi = hdi(np.asarray(draws, dtype=float), nominal)
        if not (lo <= float(truth[name]) <= hi):
            return False
    return True


def _s8_covers(y: Array) -> bool:
    return (
        _mean_covers(y, S8_MU)
        and _sigma_chi2_covers(y - float(y.mean()), len(y) - 1, S8_SIGMA)
        and _normal_exp_reference_covers(y, {"mu": S8_MU, "sigma": S8_SIGMA})
    )


def _reject_until(
    factory: Callable[[np.random.Generator], Tuple[Array, ...]],
    ok: Callable[..., bool],
    seed_prefix: str,
    max_tries: int = 400,
) -> Tuple[Array, ...]:
    for i in range(max_tries):
        rng = np.random.default_rng(sum(map(ord, f"{seed_prefix}-try{i}")))
        draws = factory(rng)
        if ok(*draws):
            return draws
    raise RuntimeError(f"no recoverable draw for {seed_prefix}")


def write_s4() -> Path:
    def factory(rng: np.random.Generator) -> Tuple[Array, Array]:
        x = np.linspace(-1.2, 1.2, 20)
        y = S4_ALPHA + S4_BETA * x + rng.normal(0.0, S4_SIGMA, size=20)
        return x, y

    x, y = _reject_until(factory, _s4_covers, "s4-psense-v3")
    path = PACKS / "S4" / "data.csv"
    _write_xy(path, x, y)
    return path


def write_s8() -> Path:
    def factory(rng: np.random.Generator) -> Tuple[Array]:
        return (rng.normal(S8_MU, S8_SIGMA, size=30),)

    (y,) = _reject_until(factory, _s8_covers, "s8-jax-v3")
    path = PACKS / "S8" / "data.csv"
    _write_y(path, y)
    return path


def _logit_irls(
    dose: Array,
    n_trials: Array,
    deaths: Array,
) -> Tuple[Array, Array] | None:
    """Grouped-binomial logistic MLE and covariance, or None if singular."""
    design = np.column_stack([np.ones(len(dose)), dose])
    beta = np.zeros(2)
    for _ in range(25):
        eta = design @ beta
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
        prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
        weight = n_trials * prob * (1.0 - prob)
        working = eta + (deaths / n_trials - prob) / (prob * (1.0 - prob))
        xtw = design.T * weight
        try:
            nxt = np.linalg.solve(xtw @ design, xtw @ working)
        except np.linalg.LinAlgError:
            return None
        if float(np.max(np.abs(nxt - beta))) < 1e-8:
            beta = nxt
            break
        beta = nxt
    eta = design @ beta
    prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -20.0, 20.0)))
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
    weight = n_trials * prob * (1.0 - prob)
    try:
        cov = np.linalg.inv((design.T * weight) @ design)
    except np.linalg.LinAlgError:
        return None
    return beta, cov


def _bioassay_covers(dose: Array, n_trials: Array, deaths: Array) -> bool:
    fit = _logit_irls(dose, n_trials, deaths)
    if fit is None:
        return False
    hat, cov = fit
    z = _z_nominal(0.94)
    if abs(float(hat[1]) - S6_BETA) > z * float(np.sqrt(cov[1, 1])):
        return False
    ld50 = -float(hat[0]) / float(hat[1])
    grad = np.array([-1.0 / float(hat[1]), float(hat[0]) / float(hat[1]) ** 2])
    se_ld = float(np.sqrt(grad @ cov @ grad))
    return abs(ld50 - S6_LD50) <= z * se_ld


def write_s6() -> Path:
    n_each = 24
    doses = np.linspace(-1.2, 1.2, 6)

    def factory(rng: np.random.Generator) -> Tuple[Array, Array, Array]:
        eta = S6_ALPHA + S6_BETA * doses
        prob = 1.0 / (1.0 + np.exp(-eta))
        deaths = rng.binomial(n_each, prob)
        return doses, np.full(len(doses), n_each, dtype=float), deaths.astype(float)

    dose, n_trials, deaths = _reject_until(factory, _bioassay_covers, "s6-bioassay-v1")
    path = PACKS / "S6" / "data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dose", "n", "y"])
        for xi, ni, yi in zip(dose, n_trials, deaths):
            writer.writerow([f"{float(xi):.6f}", int(ni), int(yi)])
    return path


def _em_two_normal(y: Array) -> Tuple[float, float, float]:
    """Ordered two-component normal EM. Returns weight, mu1, mu2."""
    mu1, mu2 = float(np.quantile(y, 0.25)), float(np.quantile(y, 0.75))
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
    weight = 0.5
    sigma = max(float(y.std(ddof=1)), 1e-3)
    for _ in range(40):
        e1 = np.exp(-0.5 * ((y - mu1) / sigma) ** 2)
        e2 = np.exp(-0.5 * ((y - mu2) / sigma) ** 2)
        r1 = weight * e1
        r2 = (1.0 - weight) * e2
        den = r1 + r2 + 1e-12
        g1 = r1 / den
        g2 = r2 / den
        weight = float(g1.mean())
        mu1 = float((g1 * y).sum() / (g1.sum() + 1e-12))
        mu2 = float((g2 * y).sum() / (g2.sum() + 1e-12))
        if mu1 > mu2:
            mu1, mu2 = mu2, mu1
            weight = 1.0 - weight
            g1, g2 = g2, g1
        var = float((g1 * (y - mu1) ** 2 + g2 * (y - mu2) ** 2).mean())
        sigma = max(float(np.sqrt(var)), 1e-3)
    return weight, mu1, mu2


def _s7_covers(y: Array) -> bool:
    weight, mu1, mu2 = _em_two_normal(y)
    if not (
        abs(mu1 - S7_MU1) < 0.35
        and abs(mu2 - S7_MU2) < 0.35
        and abs(weight - S7_WEIGHT) < 0.12
    ):
        return False
    rng = np.random.default_rng(0)
    boots = np.empty((80, 3), dtype=float)
    for i in range(80):
        samp = rng.choice(y, size=len(y), replace=True)
        boots[i] = _em_two_normal(samp)
    checks = (
        (boots[:, 0], S7_WEIGHT),
        (boots[:, 1], S7_MU1),
        (boots[:, 2], S7_MU2),
    )
    for draws, truth in checks:
        lo, hi = hdi(draws, 0.94)
        if not (lo <= truth <= hi):
            return False
    return True


def write_s7() -> Path:
    def factory(rng: np.random.Generator) -> Tuple[Array]:
        n = 90
        z = rng.random(n) < S7_WEIGHT
        y = np.empty(n)
        y[z] = rng.normal(S7_MU1, S7_SIGMA, size=int(z.sum()))
        y[~z] = rng.normal(S7_MU2, S7_SIGMA, size=int((~z).sum()))
        return (y,)

    (y,) = _reject_until(factory, _s7_covers, "s7-mixture-v2")
    path = PACKS / "S7" / "data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_y(path, y)
    return path


def _read_s6(path: Path) -> Tuple[Array, Array, Array]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    dose = np.asarray([float(r["dose"]) for r in rows], dtype=float)
    n_trials = np.asarray([float(r["n"]) for r in rows], dtype=float)
    deaths = np.asarray([float(r["y"]) for r in rows], dtype=float)
    return dose, n_trials, deaths


def check_s4() -> bool:
    path = PACKS / "S4" / "data.csv"
    if not path.is_file():
        return False
    x, y = _read_xy(path)
    return _s4_covers(x, y)


def check_s8() -> bool:
    path = PACKS / "S8" / "data.csv"
    if not path.is_file():
        return False
    return _s8_covers(_read_y(path))


def check_s6() -> bool:
    path = PACKS / "S6" / "data.csv"
    if not path.is_file():
        return False
    return _bioassay_covers(*_read_s6(path))


def check_s7() -> bool:
    path = PACKS / "S7" / "data.csv"
    if not path.is_file():
        return False
    return _s7_covers(_read_y(path))


CHECKERS: Dict[str, Callable[[], bool]] = {
    "S4": check_s4,
    "S8": check_s8,
    "S6": check_s6,
    "S7": check_s7,
}
WRITERS: Dict[str, Callable[[], Path]] = {
    "S4": write_s4,
    "S8": write_s8,
    "S6": write_s6,
    "S7": write_s7,
}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", nargs="+", default=["S4", "S8", "S6", "S7"])
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate existing CSVs; do not write",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite even if the current CSV already passes",
    )
    args = parser.parse_args(argv)
    failed = 0
    for pack_id in args.packs:
        if pack_id not in WRITERS:
            raise SystemExit(f"refuses to regenerate {pack_id} (would void live scores)")
        ok = CHECKERS[pack_id]()
        if args.check:
            print(f"{pack_id} recoverable={ok}")
            failed += int(not ok)
            continue
        if ok and not args.force:
            print(f"{pack_id} already recoverable; left in place")
            continue
        WRITERS[pack_id]()
        print(f"wrote {pack_id}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
