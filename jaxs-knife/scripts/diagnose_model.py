"""Convergence, LOO, and PPC presence checks on ArviZ InferenceData.

No PyMC dependency. LOO requires a ``log_likelihood`` group already on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    import arviz as az
except ImportError as exc:  # pragma: no cover
    raise ImportError("arviz is required for diagnose_model") from exc

from json_util import json_default

try:
    import arviz_stats as azs

    HAS_DIAGNOSE = hasattr(azs, "diagnose")
except ImportError:
    HAS_DIAGNOSE = False


# ==================================================
# Group / LOO helpers
# ==================================================


def group_names(idata: Any) -> Set[str]:
    """Bare InferenceData / DataTree group names (no leading slash)."""
    groups = getattr(idata, "groups", None)
    if groups is None:
        return set()
    raw = groups() if callable(groups) else groups
    return {str(g).strip("/") for g in raw}


def _loo_field(loo: Any, *names: str) -> Any:
    """First present ELPDData attribute among ``names`` (ArviZ 0.23 and 1.x)."""
    for name in names:
        if hasattr(loo, name):
            return getattr(loo, name)
    return None


def _max_from_dataset(ds: Any) -> Optional[float]:
    """Scalar max from an xarray Dataset of diagnostic values."""
    if ds is None:
        return None
    try:
        data_vars = getattr(ds, "data_vars", None)
        if data_vars is not None:
            vals = [float(ds[v].max()) for v in data_vars]
        else:
            vals = [float(ds.max())]
        finite = [v for v in vals if np.isfinite(v)]
        return max(finite) if finite else None
    except Exception:
        return None


# ==================================================
# Checks
# ==================================================


_DIVERGENCE_NAMES = ("diverging", "is_divergent", "divergent")


def _divergence_block(idata: Any) -> Dict[str, Any]:
    """Read sampler divergences. A missing flag is unknown, not zero."""
    sample_stats = getattr(idata, "sample_stats", None)
    data_vars = getattr(sample_stats, "data_vars", None) if sample_stats is not None else None
    if data_vars is None:
        return {
            "recorded": False,
            "count": None,
            "pct": None,
            "ok": False,
            "message": (
                "No sample_stats group. Pass diverging (BlackJAX is_divergent "
                "is renamed on conversion). Missing is not a pass."
            ),
        }
    name = next((n for n in _DIVERGENCE_NAMES if n in data_vars), None)
    if name is None:
        return {
            "recorded": False,
            "count": None,
            "pct": None,
            "ok": False,
            "message": (
                "sample_stats has no diverging / is_divergent flag. "
                "Missing is not a pass."
            ),
        }
    arr = sample_stats[name]
    n_div = int(arr.sum())
    total = int(arr.size)
    return {
        "recorded": True,
        "count": n_div,
        "pct": round(100 * n_div / total, 2) if total else 0.0,
        "ok": n_div == 0,
    }


def _parameter_names(idata: Any) -> List[str]:
    """Posterior names that are parameters, not generated quantities."""
    data_vars = getattr(idata.posterior, "data_vars", [])
    skip_exact = {"lp__", "energy__"}
    skip_prefixes = ("y_rep", "yrep", "log_lik", "log_likelihood")
    keep: List[str] = []
    for name in data_vars:
        low = str(name).lower()
        if low in skip_exact:
            continue
        if any(low == p or low.startswith(p) for p in skip_prefixes):
            continue
        keep.append(str(name))
    return keep


def check_convergence(
    idata: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """R-hat, ESS, divergences, and (when available) E-BFMI / treedepth.

    Parameters
    ----------
    idata : Any
        ArviZ InferenceData or DataTree.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Any]
        Serializable convergence block for ``check_diagnostics``.
    """
    divergences = _divergence_block(idata)
    if HAS_DIAGNOSE:
        _has_errors, diag = azs.diagnose(
            idata, return_diagnostics=True, show_diagnostics=False
        )
        skip = ("y_rep", "yrep", "log_lik", "log_likelihood", "lp__")
        rhat_bad = [
            str(p)
            for p in diag.get("rhat", {}).get("bad_params", [])
            if not str(p).lower().startswith(skip)
        ]
        ess_bad = [
            str(p)
            for p in diag.get("ess", {}).get("bad_params", [])
            if not str(p).lower().startswith(skip)
        ]
        td = diag.get("treedepth", {})
        n_td = int(td.get("n_max", 0))
        failed_chains = [int(c) for c in diag.get("bfmi", {}).get("failed_chains", [])]
        all_ok = (
            len(rhat_bad) == 0
            and len(ess_bad) == 0
            and bool(divergences.get("ok"))
            and n_td == 0
            and len(failed_chains) == 0
        )
        if logger is not None:
            logger.info("Convergence via arviz_stats.diagnose; all_ok=%s", all_ok)
        return {
            "all_ok": all_ok,
            "method": "arviz_stats.diagnose",
            "rhat": {
                "ok": len(rhat_bad) == 0,
                "max": _max_from_dataset(diag.get("rhat", {}).get("rhat_values")),
                "problematic_params": rhat_bad,
            },
            "ess_bulk": {"ok": len(ess_bad) == 0, "problematic_params": ess_bad},
            "ess_tail": {"ok": len(ess_bad) == 0, "problematic_params": ess_bad},
            "divergences": divergences,
            "treedepth": {
                "ok": n_td == 0,
                "n_max": n_td,
                "pct": round(float(td.get("pct", 0.0)), 2),
            },
            "bfmi": {"ok": len(failed_chains) == 0, "failed_chains": failed_chains},
        }

    try:
        param_names = _parameter_names(idata)
        summary = az.summary(idata, var_names=param_names) if param_names else az.summary(idata)
        posterior = idata.posterior
        sizes = getattr(posterior, "sizes", None)
        if sizes is not None and "chain" in sizes:
            n_chains = int(sizes["chain"])
        else:
            n_chains = int(posterior.chain.size)
        rhat = summary["r_hat"]
        rhat_ok = bool(rhat.notna().all() and (rhat <= 1.01).all())
        ess_bulk_ok = bool((summary["ess_bulk"] >= 100 * n_chains).all())
        ess_tail_ok = bool((summary["ess_tail"] >= 100 * n_chains).all())

        results: Dict[str, Any] = {
            "rhat": {
                "max": float(rhat.max()) if rhat.notna().any() else None,
                "ok": rhat_ok,
                "problematic_params": list(summary[(rhat > 1.01) | rhat.isna()].index),
            },
            "ess_bulk": {
                "min": int(summary["ess_bulk"].min()),
                "ok": ess_bulk_ok,
                "problematic_params": list(
                    summary[summary["ess_bulk"] < 100 * n_chains].index
                ),
            },
            "ess_tail": {
                "min": int(summary["ess_tail"].min()),
                "ok": ess_tail_ok,
                "problematic_params": list(
                    summary[summary["ess_tail"] < 100 * n_chains].index
                ),
            },
            "divergences": divergences,
            "method": "manual",
        }
        results["all_ok"] = all(
            results[k]["ok"] for k in ("rhat", "ess_bulk", "ess_tail", "divergences")
        )
        if logger is not None:
            logger.info("Convergence via az.summary; all_ok=%s", results["all_ok"])
        return results
    except Exception as exc:
        if logger is not None:
            logger.exception("az.summary fallback failed")
        return {
            "all_ok": False,
            "method": "manual",
            "error": str(exc),
            "rhat": {
                "ok": False,
                "max": None,
                "problematic_params": ["diagnose_fallback_failed"],
            },
            "ess_bulk": {"ok": False, "problematic_params": ["diagnose_fallback_failed"]},
            "ess_tail": {"ok": False, "problematic_params": ["diagnose_fallback_failed"]},
            "divergences": {
                "recorded": False,
                "count": None,
                "pct": None,
                "ok": False,
            },
        }


def check_loo(
    idata: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """PSIS-LOO and Pareto-k. Requires ``log_likelihood`` on ``idata``.

    Parameters
    ----------
    idata : Any
        ArviZ InferenceData with a log_likelihood group.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Any]
        Serializable LOO block.
    """
    if "log_likelihood" not in group_names(idata):
        if logger is not None:
            logger.warning("LOO skipped: no log_likelihood group")
        return {
            "computed": False,
            "error": (
                "No log_likelihood group. Store pointwise log likelihood in "
                "generated quantities (Stan) or vmap (JAX) before saving."
            ),
        }
    try:
        loo = az.loo(idata, pointwise=True)
        pareto_k = np.asarray(_loo_field(loo, "pareto_k"), dtype=float)
        elpd = _loo_field(loo, "elpd_loo", "elpd")
        se = _loo_field(loo, "se")
        p_loo = _loo_field(loo, "p_loo", "p")
        finite = pareto_k[np.isfinite(pareto_k)]
        n_nonfinite = int(pareto_k.size - finite.size)
        n_high = int(np.sum(finite > 0.7))
        if logger is not None:
            logger.info("LOO computed; n_bad=%s n_nonfinite=%s", n_high, n_nonfinite)
        return {
            "elpd": float(elpd) if elpd is not None else None,
            "se": float(se) if se is not None else None,
            "p_loo": float(p_loo) if p_loo is not None else None,
            "pareto_k": {
                "max": float(finite.max()) if finite.size else None,
                "n_bad": n_high,
                "n_marginal": int(np.sum((finite > 0.5) & (finite <= 0.7))),
                "n_nonfinite": n_nonfinite,
                "ok": n_high == 0 and n_nonfinite == 0,
            },
            "computed": True,
        }
    except Exception as exc:
        if logger is not None:
            logger.exception("LOO failed")
        return {"computed": False, "error": str(exc)}


def check_posterior_predictive(idata: Any) -> Dict[str, Any]:
    """Whether posterior predictive draws exist (Stan GQ or JAX vmap)."""
    if "posterior_predictive" in group_names(idata):
        pp_vars = list(idata.posterior_predictive.data_vars)
    else:
        gq = [
            n
            for n in getattr(idata.posterior, "data_vars", [])
            if str(n).lower().startswith("y_rep") or str(n).lower().startswith("yrep")
        ]
        if not gq:
            return {
                "available": False,
                "message": (
                    "No posterior predictive samples. Use generated quantities (Stan) "
                    "or vmap the observation model (JAX); do not rewrite the likelihood in numpy."
                ),
            }
        return {
            "available": True,
            "variables": gq,
            "note": "y_rep still in posterior; pass it as posterior_predictive",
        }
    results: Dict[str, Any] = {"available": True, "variables": pp_vars}
    if hasattr(idata, "observed_data"):
        results["observed_variables"] = list(idata.observed_data.data_vars)
    return results


def generate_report(
    idata: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Full diagnostics report from InferenceData.

    Parameters
    ----------
    idata : Any
        ArviZ InferenceData or DataTree.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Any]
        ``convergence``, ``loo``, ``posterior_predictive``, ``overall``.
    """
    report: Dict[str, Any] = {
        "convergence": check_convergence(idata, logger=logger),
        "loo": check_loo(idata, logger=logger),
        "posterior_predictive": check_posterior_predictive(idata),
    }
    issues: List[str] = []
    if not report["convergence"]["all_ok"]:
        issues.append("Convergence issues detected — do not interpret the posterior")
    loo_pk = report["loo"].get("pareto_k", {}) if report["loo"].get("computed") else {}
    if loo_pk and not loo_pk.get("ok", True):
        n_high = loo_pk.get("n_bad", 0)
        n_nf = loo_pk.get("n_nonfinite", 0)
        parts: List[str] = []
        if n_high:
            parts.append(f"{n_high} observation(s) with Pareto k > 0.7")
        if n_nf:
            parts.append(f"{n_nf} observation(s) with non-finite Pareto k")
        issues.append("; ".join(parts) if parts else "influential observations in LOO")
    if not report["posterior_predictive"]["available"]:
        issues.append("No posterior predictive checks available")
    report["overall"] = {
        "ok": len(issues) == 0,
        "issues": issues,
        "recommendation": (
            "Model is ready for interpretation."
            if not issues
            else "Address before interpreting: " + "; ".join(issues)
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian model diagnostics")
    parser.add_argument("--idata", required=True, help="Path to InferenceData (.nc)")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    try:
        idata = az.from_netcdf(args.idata)
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": f"Could not load InferenceData: {exc}"}))
        sys.exit(1)

    report = generate_report(idata)
    output = json.dumps(report, indent=2, default=json_default)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
