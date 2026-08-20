"""Band B: does a nominal HDI contain known parameter values?"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

DEFAULT_ALIASES: Dict[str, tuple[str, ...]] = {
    "ld50": ("ld50", "LD50", "ld_50", "dose50", "x50"),
    "mu1": ("mu1", "mu_1", "mu[1]"),
    "mu2": ("mu2", "mu_2", "mu[2]"),
    "weight": ("weight", "theta", "pi", "lambda", "mix_weight"),
    "alpha": ("alpha", "a", "intercept"),
    "beta": ("beta", "b", "slope"),
    "sigma": ("sigma", "sig", "scale"),
    "mu": ("mu", "mean"),
    "tau": ("tau", "sigma_group", "sd_group", "group_sd"),
    "theta1": ("theta1", "theta_1", "theta[1]"),
    "prevalence": ("prevalence", "p", "infection_rate"),
}


def hdi(draws: NDArray[np.floating], prob: float) -> tuple[float, float]:
    """Lowest-width interval containing ``prob`` of ``draws``.

    Parameters
    ----------
    draws : NDArray
        Posterior draws, any shape (raveled).
    prob : float
        Nominal probability in (0, 1).

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)``.
    """
    x = np.sort(np.asarray(draws, dtype=float).ravel())
    n = int(x.size)
    n_in = max(int(np.ceil(prob * n)), 1)
    widths = x[n_in - 1 :] - x[: n - n_in + 1]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + n_in - 1])


def assess_recovery(
    posterior: Mapping[str, NDArray[np.floating]],
    truth: Mapping[str, float],
    *,
    nominal: float = 0.94,
) -> Dict[str, Any]:
    """Check that each true value lies in the nominal HDI.

    Parameters
    ----------
    posterior : Mapping[str, NDArray]
        Draw arrays keyed by parameter name.
    truth : Mapping[str, float]
        True values for a subset of those names.
    nominal : float
        HDI probability.

    Returns
    -------
    Dict[str, Any]
        Per-parameter intervals and an overall ``passed`` flag.
    """
    rows: Dict[str, Any] = {}
    all_ok = True
    for name, value in truth.items():
        if name not in posterior:
            rows[name] = {"ok": False, "error": "parameter missing"}
            all_ok = False
            continue
        lo, hi = hdi(np.asarray(posterior[name]), nominal)
        ok = lo <= float(value) <= hi
        rows[name] = {
            "truth": float(value),
            "hdi": [lo, hi],
            "nominal": nominal,
            "ok": ok,
        }
        all_ok = all_ok and ok
    return {"parameters": rows, "passed": all_ok, "nominal": nominal}


def _var_names(post: Any) -> list[str]:
    data_vars = getattr(post, "data_vars", None)
    if data_vars is None:
        return []
    return [str(name) for name in data_vars]


def _get_values(post: Any, name: str) -> Optional[NDArray[np.floating]]:
    names = _var_names(post)
    lookup = {item: item for item in names}
    lookup.update({item.lower(): item for item in names})
    key = lookup.get(name) or lookup.get(name.lower())
    if key is None:
        return None
    try:
        return np.asarray(post[key].values, dtype=float)
    except Exception:
        return None


def _vector_component(
    post: Any,
    canonical: str,
) -> Optional[NDArray[np.floating]]:
    """Map ``mu1`` / ``mu2`` / ``theta1`` onto a trailing vector axis."""
    if canonical in {"mu1", "mu2"}:
        arr = _get_values(post, "mu")
        if arr is None or arr.ndim < 1 or arr.shape[-1] != 2:
            return None
        idx = 0 if canonical == "mu1" else 1
        return arr[..., idx]
    if canonical != "theta1":
        return None
    arr = _get_values(post, "theta")
    if arr is None or arr.ndim < 1 or arr.shape[-1] < 1:
        return None
    return arr[..., 0]


def posterior_from_idata(
    idata: Any,
    names: tuple[str, ...],
    aliases: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, NDArray[np.floating]]:
    """Extract named posterior arrays. Missing names are omitted, not raised.

    Parameters
    ----------
    idata : Any
        ArviZ object with ``.posterior``.
    names : tuple[str, ...]
        Canonical parameter names (from pack ``meta.json``).
    aliases : Mapping[str, Sequence[str]], optional
        Extra names per canonical key.

    Returns
    -------
    Dict[str, NDArray]
        Draw arrays for names that were found.
    """
    out: Dict[str, NDArray[np.floating]] = {}
    try:
        post = idata.posterior
    except AttributeError:
        try:
            post = idata["posterior"]
        except Exception as exc:
            raise AttributeError("no posterior group on InferenceData") from exc
    alias_map: Dict[str, tuple[str, ...]] = dict(DEFAULT_ALIASES)
    if aliases:
        for key, values in aliases.items():
            alias_map[str(key)] = tuple(str(item) for item in values)
    for name in names:
        found = _get_values(post, name)
        if found is None:
            for alt in alias_map.get(name, ()):
                found = _get_values(post, alt)
                if found is not None:
                    break
        if found is None:
            found = _vector_component(post, name)
        if found is not None:
            out[name] = found
    return out
