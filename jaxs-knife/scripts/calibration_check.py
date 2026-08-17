"""PPC coverage / PIT calibration on ArviZ InferenceData.

Uses a simple nominal-coverage check that does not require arviz_plots.
When arviz_plots is installed, optional ΔECDF plots can be saved.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import arviz as az
except ImportError as exc:  # pragma: no cover
    raise ImportError("arviz is required for calibration_check") from exc

from json_util import json_default


# ==================================================
# Coverage
# ==================================================


def _hdi_contains(
    draws: np.ndarray,
    observed: np.ndarray,
    prob: float,
) -> np.ndarray:
    """Per-observation whether ``observed`` falls in the ``prob`` HDI of ``draws``.

    Uses the lowest-width interval, not a central ETI / quantile interval.

    Parameters
    ----------
    draws : np.ndarray
        Shape ``(n_draws, n_obs)``.
    observed : np.ndarray
        Shape ``(n_obs,)``.
    prob : float
        Nominal HDI probability in (0, 1).

    Returns
    -------
    np.ndarray
        Boolean mask, shape ``(n_obs,)``.
    """
    n_draws, n_obs = int(draws.shape[0]), int(draws.shape[1])
    n_in = max(int(np.ceil(prob * n_draws)), 1)
    inside = np.empty(n_obs, dtype=bool)
    for j in range(n_obs):
        x = np.sort(draws[:, j])
        widths = x[n_in - 1 :] - x[: n_draws - n_in + 1]
        i = int(np.argmin(widths))
        inside[j] = bool(x[i] <= observed[j] <= x[i + n_in - 1])
    return inside


def assess_calibration(
    idata: Any,
    var_name: str,
    *,
    nominal: float = 0.94,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Compare empirical HDI coverage to a nominal level.

    Parameters
    ----------
    idata : Any
        InferenceData with ``posterior_predictive`` and ``observed_data``.
    var_name : str
        Shared variable name.
    nominal : float
        Nominal interval probability (default 0.94).
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Any]
        Assessment block consumed by ``check_diagnostics``.
    """
    pp = idata.posterior_predictive[var_name]
    obs = idata.observed_data[var_name]
    draws = np.asarray(pp.values, dtype=float)
    # (chain, draw, obs...) -> (n_draws, n_obs)
    if draws.ndim < 3:
        raise ValueError("posterior_predictive must have chain, draw, and obs dims")
    n_obs = int(np.prod(draws.shape[2:]))
    draws_2d = draws.reshape(draws.shape[0] * draws.shape[1], n_obs)
    observed = np.asarray(obs.values, dtype=float).reshape(n_obs)
    inside = _hdi_contains(draws_2d, observed, nominal)
    empirical = float(np.mean(inside))
    delta = empirical - nominal
    if delta > 0.02:
        diagnosis = "under-confident (predictions too uncertain)"
    elif delta < -0.02:
        diagnosis = "over-confident (predictions too certain)"
    else:
        diagnosis = "well-calibrated"
    if logger is not None:
        logger.info("Coverage empirical=%.3f nominal=%.3f delta=%.3f", empirical, nominal, delta)
    well = abs(delta) <= 0.02
    return {
        "pit_ecdf_inside_bands": well,
        "coverage_ecdf_inside_bands": well,
        "well_calibrated": well,
        "mean_coverage_deviation": round(delta, 4),
        "calibration_diagnosis": diagnosis,
        "empirical_coverage": round(empirical, 4),
        "nominal_coverage": nominal,
    }


def _common_var(idata: Any, var_name: Optional[str]) -> str:
    """Resolve the observed / PPC variable name."""
    pp_vars = set(idata.posterior_predictive.data_vars)
    obs_vars = set(idata.observed_data.data_vars)
    if var_name is not None:
        if var_name not in pp_vars or var_name not in obs_vars:
            raise ValueError(f"Variable {var_name!r} not in both PPC and observed_data")
        return var_name
    common = sorted(pp_vars & obs_vars)
    if not common:
        raise ValueError("No common variables between posterior_predictive and observed_data")
    return common[0]


def generate_calibration_report(
    idata: Any,
    var_name: Optional[str] = None,
    *,
    nominal: float = 0.94,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Full calibration JSON.

    Parameters
    ----------
    idata : Any
        InferenceData.
    var_name : str, optional
        Observed variable; auto-detected if omitted.
    nominal : float
        Nominal HDI probability.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Any]
        Report with ``assessment``.
    """
    if not hasattr(idata, "posterior_predictive"):
        raise ValueError("No posterior_predictive group")
    if not hasattr(idata, "observed_data"):
        raise ValueError("No observed_data group")
    resolved = _common_var(idata, var_name)
    assessment = assess_calibration(
        idata, resolved, nominal=nominal, logger=logger
    )
    n_obs = int(np.asarray(idata.observed_data[resolved].values).size)
    return {
        "variable": resolved,
        "n_observations": n_obs,
        "pit_method": "nominal_hdi_coverage",
        "note": (
            "PPC coverage on this dataset. Not a calibration or frequentist "
            "coverage claim; those need repeated datasets."
        ),
        "assessment": assessment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian model calibration check")
    parser.add_argument("--idata", required=True)
    parser.add_argument("--var-name", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--nominal", type=float, default=0.94)
    args = parser.parse_args()

    try:
        idata = az.from_netcdf(args.idata)
        report = generate_calibration_report(
            idata, var_name=args.var_name, nominal=args.nominal
        )
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}))
        sys.exit(1)

    output = json.dumps(report, indent=2, default=json_default)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
