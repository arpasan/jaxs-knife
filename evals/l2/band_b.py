"""Band B: does a nominal HDI contain known parameter values?"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
from numpy.typing import NDArray


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


def posterior_from_idata(idata: Any, names: tuple[str, ...]) -> Dict[str, NDArray[np.floating]]:
    """Extract named posterior arrays from InferenceData / DataTree.

    Parameters
    ----------
    idata : Any
        ArviZ object with ``.posterior``.
    names : tuple[str, ...]
        Parameter names.

    Returns
    -------
    Dict[str, NDArray]
        Draw arrays.
    """
    out: Dict[str, NDArray[np.floating]] = {}
    post = idata.posterior
    for name in names:
        out[name] = np.asarray(post[name].values, dtype=float)
    return out
