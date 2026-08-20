"""Regenerate or screen pack CSVs. Hidden truth must be recoverable.

Each writer uses a two-sided screen: a naive interval misses the named
estimand, and a reference interval under the task's own observation
model covers it.
"""

from __future__ import annotations

import argparse
import csv
import json
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
Z_94 = 1.880793608151251

# --------------------------------------------------
# ---#---#--- Task constants ---#---#---
# --------------------------------------------------

E1_ALPHA = 0.15
E1_BETA = 1.20
E1_SIGMA = 0.28
E1_N = 80

H1_MU = 0.00
H1_TAU = 1.60
H1_SIGMA = 0.70
H1_J = 8
H1_NJ = 6

A1_P = 0.10
A1_FPR = 0.08
A1_FNR = 0.00
A1_N = 400

K1_MU1 = -1.20
K1_MU2 = 1.30
K1_WEIGHT = 0.40
K1_SIGMA = 0.45
K1_N = 110

J1_LOG_MU = 0.0
J1_LOG_SIGMA = 1.35
J1_N = 60
J1_Z95 = 1.6448536269514722
J1_Q25 = float(np.exp(-0.6744897501960817 * J1_LOG_SIGMA))
J1_Q50 = 1.0
J1_Q75 = float(np.exp(0.6744897501960817 * J1_LOG_SIGMA))
J1_Q95 = float(np.exp(J1_Z95 * J1_LOG_SIGMA))

M1_ALPHA = 0.10
M1_BETA = 1.15
M1_SIGMA = 0.40
M1_N = 100
M1_CUT = 1.15


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


# --------------------------------------------------
# ---#---#--- I/O ---#---#---
# --------------------------------------------------


def _write_y(path: Path, y: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["y"])
        for yi in y:
            writer.writerow([f"{float(yi):.6f}"])


def _write_binary_y(path: Path, y: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["y"])
        for yi in y:
            writer.writerow([int(yi)])


def _write_eiv(path: Path, x: Array, x_se: Array, y: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "x_se", "y"])
        for xi, sei, yi in zip(x, x_se, y):
            writer.writerow(
                [f"{float(xi):.6f}", f"{float(sei):.6f}", f"{float(yi):.6f}"]
            )


def _write_group_y(path: Path, group: Array, y: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "y"])
        for gi, yi in zip(group, y):
            writer.writerow([int(gi), f"{float(yi):.6f}"])


def _read_y(path: Path) -> Array:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return np.asarray([float(r["y"]) for r in rows], dtype=float)


def _read_eiv(path: Path) -> Tuple[Array, Array, Array]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    x = np.asarray([float(r["x"]) for r in rows], dtype=float)
    x_se = np.asarray([float(r["x_se"]) for r in rows], dtype=float)
    y = np.asarray([float(r["y"]) for r in rows], dtype=float)
    return x, x_se, y


def _read_group_y(path: Path) -> Tuple[Array, Array]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    group = np.asarray([float(r["group"]) for r in rows], dtype=float)
    y = np.asarray([float(r["y"]) for r in rows], dtype=float)
    return group, y


def _write_xy_maybe_y(path: Path, x: Array, y: Array, observed: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        for xi, yi, keep in zip(x, y, observed):
            if bool(keep):
                writer.writerow([f"{float(xi):.6f}", f"{float(yi):.6f}"])
            else:
                writer.writerow([f"{float(xi):.6f}", ""])


def _read_xy_maybe_y(path: Path) -> Tuple[Array, Array, Array]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    x = np.asarray([float(r["x"]) for r in rows], dtype=float)
    y_raw = [r.get("y", "") for r in rows]
    observed = np.asarray([str(v).strip() != "" for v in y_raw], dtype=bool)
    y = np.full(len(rows), np.nan, dtype=float)
    for i, raw in enumerate(y_raw):
        if observed[i]:
            y[i] = float(raw)
    return x, y, observed


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


# --------------------------------------------------
# ---#---#--- Shared interval helpers ---#---#---
# --------------------------------------------------


def _ols_beta_covers(
    x: Array,
    y: Array,
    beta: float,
    *,
    nominal: float = 0.94,
) -> bool:
    """True when a normal-approx OLS interval contains the slope."""
    design = np.column_stack([np.ones(len(x)), x])
    hat, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ hat
    n, p = design.shape
    cov = (resid.dot(resid) / (n - p)) * np.linalg.inv(design.T @ design)
    se = float(np.sqrt(cov[1, 1]))
    return bool(abs(float(hat[1]) - beta) <= _z_nominal(nominal) * se)


def _mean_covers(y: Array, truth: float, nominal: float = 0.94) -> bool:
    se = float(y.std(ddof=1) / np.sqrt(len(y)))
    return bool(abs(float(y.mean()) - truth) <= _z_nominal(nominal) * se)


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


def _hdi_from_grid(logp: Array, coords: Tuple[Array, ...], n_draw: int) -> List[Array]:
    """Draw from a normalized grid and return one array per axis."""
    logp = np.asarray(logp, dtype=float)
    logp = logp - float(logp.max())
    weight = np.exp(logp)
    weight /= float(weight.sum())
    rng = np.random.default_rng(0)
    flat = weight.ravel()
    idx = rng.choice(flat.size, size=n_draw, p=flat)
    out: List[Array] = []
    remainder = idx
    for axis, coord in enumerate(coords):
        stride = 1
        for later in coords[axis + 1 :]:
            stride *= int(later.size)
        take = remainder // stride
        remainder = remainder % stride
        out.append(coord[take])
    return out


# ==================================================
# E1 — errors in the predictor
# ==================================================


def _eiv_latent_moments(x_obs: Array, x_se: Array) -> Tuple[float, float]:
    """Mean and variance of unmarked x from the instrument columns only.

    A solver estimates these. Do not pin them at generating values.
    """
    x_obs = np.asarray(x_obs, dtype=float)
    x_se = np.asarray(x_se, dtype=float)
    mu_x = float(x_obs.mean())
    var_obs = float(x_obs.var(ddof=1)) if x_obs.size > 1 else 0.0
    mean_se2 = float(np.mean(np.square(x_se)))
    var_x = max(var_obs - mean_se2, 1e-4)
    return mu_x, var_x


def _eiv_reference_covers(
    x_obs: Array,
    x_se: Array,
    y: Array,
    truth: Mapping[str, float],
    *,
    nominal: float = 0.94,
) -> bool:
    """Grid posterior with latent-x mean and variance estimated from the file."""
    x_obs = np.asarray(x_obs, dtype=float)
    x_se = np.asarray(x_se, dtype=float)
    y = np.asarray(y, dtype=float)
    alphas = np.linspace(-1.2, 1.6, 71)
    betas = np.linspace(-0.1, 2.5, 81)
    sigmas = np.linspace(0.10, 1.05, 45)
    mu_x, var_x = _eiv_latent_moments(x_obs, x_se)
    var_xo = var_x + x_se**2
    dx = x_obs - mu_x
    dx2 = dx**2
    two_pi = 2.0 * np.pi
    logp = np.empty((alphas.size, betas.size, sigmas.size), dtype=float)
    for ib, beta in enumerate(betas):
        cov = beta * var_x
        my0 = beta * mu_x
        for is_, sigma in enumerate(sigmas):
            var_y = beta * beta * var_x + sigma * sigma
            det = np.clip(var_y * var_xo - cov * cov, 1e-12, None)
            dy = y[None, :] - alphas[:, None] - my0
            quad = (var_xo * dy**2 - 2.0 * cov * dy * dx + var_y * dx2) / det
            ll = -np.log(two_pi) - 0.5 * np.log(det) - 0.5 * quad
            logp[:, ib, is_] = (
                ll.sum(axis=1)
                - 0.5 * (alphas / 2.5) ** 2
                - 0.5 * (beta / 2.5) ** 2
                - sigma
            )
    a_s, b_s, _s_s = _hdi_from_grid(logp, (alphas, betas, sigmas), 8000)
    for name, draws in (("alpha", a_s), ("beta", b_s)):
        lo, hi = hdi(np.asarray(draws, dtype=float), nominal)
        if not (lo <= float(truth[name]) <= hi):
            return False
    return True


def _e1_covers(x: Array, x_se: Array, y: Array) -> bool:
    """OLS on printed x misses beta; latent-x grid covers alpha and beta."""
    if _ols_beta_covers(x, y, E1_BETA):
        return False
    return _eiv_reference_covers(
        x, x_se, y, {"alpha": E1_ALPHA, "beta": E1_BETA}
    )


def write_e1() -> Path:
    def factory(rng: np.random.Generator) -> Tuple[Array, Array, Array]:
        x_true = rng.normal(0.0, 1.0, size=E1_N)
        x_se = rng.uniform(0.55, 0.85, size=E1_N)
        x_obs = x_true + rng.normal(0.0, x_se)
        y = E1_ALPHA + E1_BETA * x_true + rng.normal(0.0, E1_SIGMA, size=E1_N)
        return x_obs, x_se, y

    x, x_se, y = _reject_until(factory, _e1_covers, "e1-eiv-v2")
    path = PACKS / "E1" / "data.csv"
    _write_eiv(path, x, x_se, y)
    return path


def check_e1() -> bool:
    path = PACKS / "E1" / "data.csv"
    if not path.is_file():
        return False
    return _e1_covers(*_read_eiv(path))


# ==================================================
# H1 — grouped observations
# ==================================================


def _hier_reference_covers(
    group: Array,
    y: Array,
    truth: Mapping[str, float],
    *,
    nominal: float = 0.94,
) -> bool:
    """Bootstrap MOM + shrinkage for mu, tau, theta1; new group from N(mu, tau)."""
    group = np.asarray(group, dtype=int)
    y = np.asarray(y, dtype=float)
    groups = np.unique(group)
    if int(groups[0]) != 1:
        return False
    rng = np.random.default_rng(0)
    n_boot = 120
    mus = np.empty(n_boot)
    taus = np.empty(n_boot)
    t1s = np.empty(n_boot)
    tnews = np.empty(n_boot)
    for i in range(n_boot):
        bars: List[float] = []
        within_num = 0.0
        within_df = 0
        n1 = 0
        for g in groups:
            yg = y[group == g]
            samp = rng.choice(yg, size=len(yg), replace=True)
            bars.append(float(samp.mean()))
            within_num += float(np.sum((samp - samp.mean()) ** 2))
            within_df += int(len(samp) - 1)
            if int(g) == 1:
                n1 = int(len(samp))
        bars_a = np.asarray(bars, dtype=float)
        mu = float(bars_a.mean())
        within = within_num / max(within_df, 1)
        between = float(bars_a.var(ddof=1))
        tau2 = max(between - within / float(H1_NJ), 1e-8)
        se2 = within / max(n1, 1)
        weight = tau2 / (tau2 + se2)
        mus[i] = mu
        taus[i] = float(np.sqrt(tau2))
        t1s[i] = weight * bars_a[0] + (1.0 - weight) * mu
        tnews[i] = mu + taus[i] * float(rng.normal())
    draws_by = {"mu": mus, "tau": taus, "theta1": t1s, "theta_new": tnews}
    for key, value in truth.items():
        if key not in draws_by:
            return False
        lo, hi = hdi(np.asarray(draws_by[key], dtype=float), nominal)
        if not (lo <= float(value) <= hi):
            return False
    return True


def _h1_truth() -> Dict[str, float]:
    meta_path = PACKS / "H1" / "meta.json"
    if not meta_path.is_file():
        return {"mu": H1_MU, "tau": H1_TAU}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = dict(meta.get("truth") or {})
    return {str(k): float(v) for k, v in raw.items()}


def _h1_covers(group: Array, y: Array, theta1: float, theta_new: float) -> bool:
    """Complete-pool misses theta1 and theta_new; hierarchical reference covers."""
    if _mean_covers(y, theta1) or _mean_covers(y, theta_new):
        return False
    return _hier_reference_covers(
        group,
        y,
        {"mu": H1_MU, "tau": H1_TAU, "theta1": theta1, "theta_new": theta_new},
    )


def write_h1() -> Path:
    def factory(rng: np.random.Generator) -> Tuple[Array, Array, Array, Array]:
        theta = rng.normal(H1_MU, H1_TAU, size=H1_J)
        order = np.argsort(-np.abs(theta))
        theta = theta[order]
        group = np.repeat(np.arange(1, H1_J + 1), H1_NJ).astype(float)
        y = np.empty(H1_J * H1_NJ)
        for j in range(H1_J):
            sl = slice(j * H1_NJ, (j + 1) * H1_NJ)
            y[sl] = theta[j] + rng.normal(0.0, H1_SIGMA, size=H1_NJ)
        theta_new = rng.normal(H1_MU, H1_TAU, size=1)
        return group, y, theta, theta_new

    def ok(group: Array, y: Array, theta: Array, theta_new: Array) -> bool:
        return _h1_covers(group, y, float(theta[0]), float(theta_new[0]))

    group, y, theta, theta_new = _reject_until(factory, ok, "h1-group-v3")
    path = PACKS / "H1" / "data.csv"
    _write_group_y(path, group, y)
    meta_path = PACKS / "H1" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["truth"] = {
        "mu": H1_MU,
        "tau": H1_TAU,
        "theta1": round(float(theta[0]), 6),
        "theta_new": round(float(theta_new[0]), 6),
    }
    meta["aliases"] = {
        **dict(meta.get("aliases") or {}),
        "theta_new": ["theta_new", "theta_pred", "theta_star", "new_theta"],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def check_h1() -> bool:
    path = PACKS / "H1" / "data.csv"
    if not path.is_file():
        return False
    truth = _h1_truth()
    if "theta1" not in truth or "theta_new" not in truth:
        return False
    return _h1_covers(
        *_read_group_y(path),
        float(truth["theta1"]),
        float(truth["theta_new"]),
    )


# ==================================================
# A1 — imperfect assay
# ==================================================


def _naive_prop_covers(y: Array, truth: float, nominal: float = 0.94) -> bool:
    n = int(y.size)
    phat = float(y.mean())
    se = float(np.sqrt(max(phat * (1.0 - phat), 1e-12) / n))
    return bool(abs(phat - truth) <= _z_nominal(nominal) * se)


def _assay_reference_covers(
    y: Array,
    truth: float,
    *,
    fpr: float = A1_FPR,
    fnr: float = A1_FNR,
    nominal: float = 0.94,
) -> bool:
    """Grid posterior for prevalence given a stated false-positive rate."""
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    k = float(y.sum())
    ps = np.linspace(1e-4, 0.80, 401)
    p_obs = ps * (1.0 - fnr) + (1.0 - ps) * fpr
    p_obs = np.clip(p_obs, 1e-8, 1.0 - 1e-8)
    logp = (
        k * np.log(p_obs)
        + (n - k) * np.log(1.0 - p_obs)
        + np.log(ps)
        + np.log(1.0 - ps)
    )
    (draws,) = _hdi_from_grid(logp, (ps,), 6000)
    lo, hi = hdi(np.asarray(draws, dtype=float), nominal)
    return bool(lo <= truth <= hi)


def _a1_covers(y: Array) -> bool:
    """Naive binomial interval misses prevalence; assay-corrected grid covers."""
    if _naive_prop_covers(y, A1_P):
        return False
    return _assay_reference_covers(y, A1_P)


def write_a1() -> Path:
    def factory(rng: np.random.Generator) -> Tuple[Array]:
        z = rng.random(A1_N) < A1_P
        p_call = np.where(z, 1.0 - A1_FNR, A1_FPR)
        y = (rng.random(A1_N) < p_call).astype(float)
        return (y,)

    (y,) = _reject_until(factory, _a1_covers, "a1-assay-v1")
    path = PACKS / "A1" / "data.csv"
    _write_binary_y(path, y)
    return path


def check_a1() -> bool:
    path = PACKS / "A1" / "data.csv"
    if not path.is_file():
        return False
    return _a1_covers(_read_y(path))


# ==================================================
# K1 — two-component sample
# ==================================================


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


def _k1_covers(y: Array) -> bool:
    """Single-normal mean misses both locations; EM bootstrap covers all three."""
    if _mean_covers(y, K1_MU1) or _mean_covers(y, K1_MU2):
        return False
    weight, mu1, mu2 = _em_two_normal(y)
    if not (
        abs(mu1 - K1_MU1) < 0.40
        and abs(mu2 - K1_MU2) < 0.40
        and abs(weight - K1_WEIGHT) < 0.14
    ):
        return False
    rng = np.random.default_rng(0)
    boots = np.empty((80, 3), dtype=float)
    for i in range(80):
        samp = rng.choice(y, size=len(y), replace=True)
        boots[i] = _em_two_normal(samp)
    checks = (
        (boots[:, 0], K1_WEIGHT),
        (boots[:, 1], K1_MU1),
        (boots[:, 2], K1_MU2),
    )
    for draws, truth in checks:
        lo, hi = hdi(draws, 0.94)
        if not (lo <= truth <= hi):
            return False
    return True


def write_k1() -> Path:
    def factory(rng: np.random.Generator) -> Tuple[Array]:
        z = rng.random(K1_N) < K1_WEIGHT
        y = np.empty(K1_N)
        y[z] = rng.normal(K1_MU1, K1_SIGMA, size=int(z.sum()))
        y[~z] = rng.normal(K1_MU2, K1_SIGMA, size=int((~z).sum()))
        return (y,)

    (y,) = _reject_until(factory, _k1_covers, "k1-two-comp-v1")
    path = PACKS / "K1" / "data.csv"
    _write_y(path, y)
    return path


def check_k1() -> bool:
    path = PACKS / "K1" / "data.csv"
    if not path.is_file():
        return False
    return _k1_covers(_read_y(path))


# ==================================================
# J1 — elicited quartiles; score the 95th percentile
# ==================================================


def _bootstrap_q95(y: Array, *, log_scale: bool, seed: int) -> Array:
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    hats = np.empty(200)
    for i in range(200):
        samp = rng.choice(y, size=len(y), replace=True)
        if log_scale:
            z = np.log(np.clip(samp, 1e-8, None))
            hats[i] = float(np.exp(z.mean() + J1_Z95 * z.std(ddof=1)))
        else:
            hats[i] = float(samp.mean() + J1_Z95 * samp.std(ddof=1))
    return hats


def _j1_covers(y: Array) -> bool:
    """Gaussian q95 misses; lognormal q95 covers the process tail."""
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        return False
    gauss = _bootstrap_q95(y, log_scale=False, seed=0)
    lo_g, hi_g = hdi(gauss, 0.94)
    if lo_g <= J1_Q95 <= hi_g:
        return False
    logs = _bootstrap_q95(y, log_scale=True, seed=1)
    lo_l, hi_l = hdi(logs, 0.94)
    return bool(lo_l <= J1_Q95 <= hi_l)


def write_j1() -> Path:
    def factory(rng: np.random.Generator) -> Tuple[Array]:
        return (rng.lognormal(J1_LOG_MU, J1_LOG_SIGMA, size=J1_N),)

    (y,) = _reject_until(factory, _j1_covers, "j1-q95-v1")
    path = PACKS / "J1" / "data.csv"
    _write_y(path, y)
    meta_path = PACKS / "J1" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["truth"] = {"q95": round(J1_Q95, 6)}
    meta["aliases"] = {"q95": ["q95", "q_95", "p95", "quantile_95", "y_q95"]}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def check_j1() -> bool:
    path = PACKS / "J1" / "data.csv"
    if not path.is_file():
        return False
    return _j1_covers(_read_y(path))


# ==================================================
# M1 — blank cells (MNAR on y)
# ==================================================


def _tobit_reference_covers(
    x: Array,
    y: Array,
    observed: Array,
    truth: Mapping[str, float],
    *,
    nominal: float = 0.94,
) -> bool:
    """Grid posterior: censored-normal (n known). Not truncated-and-censored."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.asarray(observed, dtype=bool)
    y_obs = y[keep]
    x_obs = x[keep]
    x_miss = x[~keep]
    if y_obs.size < 8 or x_miss.size < 4:
        return False
    upper = float(M1_CUT)

    def log_ndtr_local(z: Array) -> Array:
        from scipy.special import erfc

        zz = np.asarray(z, dtype=float)
        return np.log(np.clip(0.5 * erfc(-zz / np.sqrt(2.0)), 1e-300, 1.0))

    alphas = np.linspace(-1.0, 1.4, 41)
    betas = np.linspace(-0.2, 2.4, 51)
    sigmas = np.linspace(0.18, 1.10, 29)
    logp = np.empty((alphas.size, betas.size, sigmas.size), dtype=float)
    for ib, beta in enumerate(betas):
        for is_, sigma in enumerate(sigmas):
            mu_obs = alphas[:, None] + beta * x_obs[None, :]
            mu_miss = alphas[:, None] + beta * x_miss[None, :]
            z_obs = (y_obs[None, :] - mu_obs) / sigma
            z_u_miss = (upper - mu_miss) / sigma
            ll = (
                -0.5 * np.log(2.0 * np.pi)
                - np.log(sigma)
                - 0.5 * z_obs**2
            ).sum(axis=1)
            ll = ll + log_ndtr_local(-z_u_miss).sum(axis=1)
            logp[:, ib, is_] = ll - 0.5 * (alphas / 2.5) ** 2 - 0.5 * (beta / 2.5) ** 2 - sigma
    a_s, b_s, _s = _hdi_from_grid(logp, (alphas, betas, sigmas), 6000)
    for name, draws in (("alpha", a_s), ("beta", b_s)):
        lo, hi = hdi(np.asarray(draws, dtype=float), nominal)
        if not (lo <= float(truth[name]) <= hi):
            return False
    return True


def _m1_covers(x: Array, y: Array, observed: Array) -> bool:
    """Complete-case OLS misses beta; censored-y reference covers alpha and beta."""
    keep = np.asarray(observed, dtype=bool)
    if _ols_beta_covers(x[keep], y[keep], M1_BETA):
        return False
    return _tobit_reference_covers(
        x, y, keep, {"alpha": M1_ALPHA, "beta": M1_BETA}
    )


def write_m1() -> Path:
    def factory(rng: np.random.Generator) -> Tuple[Array, Array, Array]:
        x = rng.normal(0.0, 1.0, size=M1_N)
        y = M1_ALPHA + M1_BETA * x + rng.normal(0.0, M1_SIGMA, size=M1_N)
        observed = y <= M1_CUT
        return x, y, observed.astype(float)

    def ok(x: Array, y: Array, observed: Array) -> bool:
        return _m1_covers(x, y, observed.astype(bool))

    x, y, observed = _reject_until(factory, ok, "m1-blank-v1")
    path = PACKS / "M1" / "data.csv"
    _write_xy_maybe_y(path, x, y, observed.astype(bool))
    meta_path = PACKS / "M1" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["truth"] = {"alpha": M1_ALPHA, "beta": M1_BETA}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def check_m1() -> bool:
    path = PACKS / "M1" / "data.csv"
    if not path.is_file():
        return False
    return _m1_covers(*_read_xy_maybe_y(path))


# --------------------------------------------------
# ---#---#--- CLI ---#---#---
# --------------------------------------------------

CHECKERS: Dict[str, Callable[[], bool]] = {
    "E1": check_e1,
    "H1": check_h1,
    "A1": check_a1,
    "K1": check_k1,
    "J1": check_j1,
    "M1": check_m1,
}
WRITERS: Dict[str, Callable[[], Path]] = {
    "E1": write_e1,
    "H1": write_h1,
    "A1": write_a1,
    "K1": write_k1,
    "J1": write_j1,
    "M1": write_m1,
}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packs",
        nargs="+",
        default=["E1", "H1", "A1", "K1", "J1", "M1"],
    )
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
        if pack_id not in CHECKERS:
            raise SystemExit(f"{pack_id} has no CSV screen")
        ok = CHECKERS[pack_id]()
        if args.check:
            print(f"{pack_id} recoverable={ok}")
            failed += int(not ok)
            continue
        if pack_id not in WRITERS:
            raise SystemExit(f"{pack_id} has no CSV writer")
        if ok and not args.force:
            print(f"{pack_id} already recoverable; left in place")
            continue
        WRITERS[pack_id]()
        print(f"wrote {pack_id}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
