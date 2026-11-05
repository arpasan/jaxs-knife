"""Convergence, LOO, and PPC presence checks on ArviZ InferenceData.

No PyMC dependency. LOO requires a ``log_likelihood`` group already on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import arviz as az
except ImportError as exc:  # pragma: no cover
    raise ImportError("arviz is required for diagnose_model") from exc

from json_util import json_default

# ==================================================
# ESS gate
# ==================================================

ESS_PER_CHAIN = 100
MIN_MCMC_CHAINS = 4
_EXEMPT_METHOD_MARKERS = ("pathfinder", "laplace", "advi", "svi")
_METHOD_ATTR_KEYS = ("inference_method", "sampler_name", "jaxs_knife_method")

azs: Any = None
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


def _n_chains(idata: Any) -> int:
    """Chain count on ``posterior``. A missing chain dim counts as 0."""
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return 0
    sizes = getattr(posterior, "sizes", None)
    if isinstance(sizes, Mapping) and "chain" in sizes:
        return int(sizes["chain"])
    chain = getattr(posterior, "chain", None)
    if chain is None:
        return 0
    try:
        return int(chain.size)
    except Exception:
        return 0


def _method_label(idata: Any, method: Optional[str]) -> str:
    """Explicit ``method`` wins; otherwise read a label stored on ``idata``.

    Unlabeled draws are NUTS/MCMC, so the chain floor applies.
    """
    if method is not None and str(method).strip():
        return str(method).strip()
    holders = (
        getattr(idata, "attrs", None),
        getattr(getattr(idata, "posterior", None), "attrs", None),
        getattr(getattr(idata, "sample_stats", None), "attrs", None),
    )
    for holder in holders:
        if not isinstance(holder, Mapping):
            continue
        for key in _METHOD_ATTR_KEYS:
            value = holder.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "nuts"


def _chain_floor_exempt(method: str) -> bool:
    """Pathfinder, VI (ADVI/SVI), and Laplace skip the NUTS chain floor."""
    text = method.strip().lower().replace("-", " ").replace("_", " ")
    if not text:
        return False
    if any(marker in text for marker in _EXEMPT_METHOD_MARKERS):
        return True
    parts = text.split()
    return "vi" in parts or "variational" in parts


def _chain_floor_status(n_chains: int, method: str) -> Dict[str, Any]:
    """NUTS/MCMC needs ``MIN_MCMC_CHAINS``. Labeled approximations are exempt."""
    exempt = _chain_floor_exempt(method)
    return {
        "minimum": MIN_MCMC_CHAINS,
        "n_chains": int(n_chains),
        "method": method,
        "exempt": exempt,
        "ok": bool(exempt or n_chains >= MIN_MCMC_CHAINS),
    }


def _apply_chain_floor(
    block: Dict[str, Any],
    idata: Any,
    method: Optional[str],
) -> Dict[str, Any]:
    """Fail the ESS gate when an MCMC fit has fewer than 4 chains."""
    label = _method_label(idata, method)
    n_chains = _n_chains(idata)
    floor = _chain_floor_status(n_chains, label)
    block["n_chains"] = n_chains
    block["sampler_method"] = label
    block["chain_floor"] = floor
    if not floor["ok"]:
        block["all_ok"] = False
        for key in ("ess_bulk", "ess_tail"):
            ess = block.get(key)
            if isinstance(ess, dict):
                ess["ok"] = False
    return block


def _as_ess_dataset(ess_result: Any) -> Any:
    """Dataset of ESS values from ``az.ess`` (Dataset or DataTree)."""
    dataset = getattr(ess_result, "dataset", None)
    if getattr(dataset, "data_vars", None) is not None:
        return dataset
    if getattr(ess_result, "data_vars", None) is not None:
        return ess_result
    raise TypeError("az.ess did not return a dataset of ESS values")


def _params_below_ess(idata: Any, *, ess_kind: str, n_chains: int) -> List[str]:
    """Parameters whose bulk or tail ESS is below 100 draws per chain."""
    names = _parameter_names(idata)
    kwargs: Dict[str, Any] = {"method": ess_kind}
    if names:
        kwargs["var_names"] = names
    dataset = _as_ess_dataset(az.ess(idata, **kwargs))
    threshold = float(ESS_PER_CHAIN * n_chains)
    skip = ("y_rep", "yrep", "log_lik", "log_likelihood", "lp__", "energy__")
    bad: List[str] = []
    for name in dataset.data_vars:
        low = str(name).lower()
        if low.startswith(skip):
            continue
        values = np.asarray(dataset[name].values, dtype=float).ravel()
        if values.size == 0 or not np.all(np.isfinite(values)) or bool(np.any(values < threshold)):
            bad.append(str(name))
    return bad


def _split_ess(
    idata: Any,
    diag: Mapping[str, Any],
    *,
    n_chains: int,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str]]:
    """Bulk and tail ESS failures, computed apart.

    A failure to compute tail ESS is not reported as a copy of the bulk list.
    """
    try:
        bulk = _params_below_ess(idata, ess_kind="bulk", n_chains=n_chains)
        tail = _params_below_ess(idata, ess_kind="tail", n_chains=n_chains)
        return bulk, tail
    except Exception:
        if logger is not None:
            logger.exception("Split ESS failed; ess_tail is not copied from ess_bulk")
        skip = ("y_rep", "yrep", "log_lik", "log_likelihood", "lp__")
        ess_block = diag.get("ess", {})
        raw = ess_block.get("bad_params", []) if isinstance(ess_block, Mapping) else []
        bulk = [
            str(name)
            for name in raw
            if not str(name).lower().startswith(skip)
        ]
        return bulk, ["ess_tail_not_computed"]


def _convergence_block(
    idata: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """R-hat, ESS, divergences, and (when available) E-BFMI / treedepth."""
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
        n_chains = _n_chains(idata)
        ess_bad, ess_tail_bad = _split_ess(idata, diag, n_chains=n_chains, logger=logger)
        td = diag.get("treedepth", {})
        n_td = int(td.get("n_max", 0))
        failed_chains = [int(c) for c in diag.get("bfmi", {}).get("failed_chains", [])]
        all_ok = (
            len(rhat_bad) == 0
            and len(ess_bad) == 0
            and len(ess_tail_bad) == 0
            and bool(divergences.get("ok"))
            and n_td == 0
            and len(failed_chains) == 0
        )
        return {
            "all_ok": all_ok,
            "method": "arviz_stats.diagnose",
            "rhat": {
                "ok": len(rhat_bad) == 0,
                "max": _max_from_dataset(diag.get("rhat", {}).get("rhat_values")),
                "problematic_params": rhat_bad,
            },
            "ess_bulk": {"ok": len(ess_bad) == 0, "problematic_params": ess_bad},
            "ess_tail": {"ok": len(ess_tail_bad) == 0, "problematic_params": ess_tail_bad},
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
        n_chains = _n_chains(idata)
        threshold = ESS_PER_CHAIN * n_chains
        rhat = summary["r_hat"]
        rhat_ok = bool(rhat.notna().all() and (rhat <= 1.01).all())
        ess_bulk_series = summary["ess_bulk"]
        ess_tail_series = summary["ess_tail"]
        ess_bulk_ok = bool((ess_bulk_series >= threshold).all())
        ess_tail_ok = bool((ess_tail_series >= threshold).all())

        results: Dict[str, Any] = {
            "rhat": {
                "max": float(rhat.max()) if rhat.notna().any() else None,
                "ok": rhat_ok,
                "problematic_params": list(summary[(rhat > 1.01) | rhat.isna()].index),
            },
            "ess_bulk": {
                "min": int(ess_bulk_series.min()),
                "ok": ess_bulk_ok,
                "problematic_params": list(summary[ess_bulk_series < threshold].index),
            },
            "ess_tail": {
                "min": int(ess_tail_series.min()),
                "ok": ess_tail_ok,
                "problematic_params": list(summary[ess_tail_series < threshold].index),
            },
            "divergences": divergences,
            "method": "manual",
        }
        results["all_ok"] = all(
            results[k]["ok"] for k in ("rhat", "ess_bulk", "ess_tail", "divergences")
        )
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


def check_convergence(
    idata: Any,
    *,
    method: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """R-hat, ESS, divergences, and (when available) E-BFMI / treedepth.

    Parameters
    ----------
    idata : Any
        ArviZ InferenceData or DataTree.
    method : str, optional
        Inference method (``nuts``, ``pathfinder``, ``vi``, ``laplace``).
        Overrides a label stored on the InferenceData. Unlabeled draws are
        treated as NUTS, which fails the ESS gate below 4 chains.
        Pathfinder, VI, and Laplace are exempt from that floor.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Any]
        Serializable convergence block for ``check_diagnostics``.
    """
    block = _apply_chain_floor(_convergence_block(idata, logger=logger), idata, method)
    if logger is not None:
        floor = block.get("chain_floor") or {}
        logger.info(
            "Convergence via %s; all_ok=%s sampler_method=%s n_chains=%s chain_floor_ok=%s",
            block.get("method"),
            block.get("all_ok"),
            block.get("sampler_method"),
            block.get("n_chains"),
            floor.get("ok") if isinstance(floor, dict) else None,
        )
    return block


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
    method: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Full diagnostics report from InferenceData.

    Parameters
    ----------
    idata : Any
        ArviZ InferenceData or DataTree.
    method : str, optional
        Inference method forwarded to ``check_convergence``. Use
        ``pathfinder``, ``vi``, or ``laplace`` to exempt an approximation
        from the NUTS chain floor. Unlabeled draws are treated as NUTS.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Any]
        ``convergence``, ``loo``, ``posterior_predictive``, ``overall``.
    """
    report: Dict[str, Any] = {
        "convergence": check_convergence(idata, method=method, logger=logger),
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
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "Inference method label (nuts, pathfinder, vi, laplace). "
            "NUTS/MCMC fails the ESS gate when n_chains < 4."
        ),
    )
    args = parser.parse_args()

    try:
        idata = az.from_netcdf(args.idata)
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": f"Could not load InferenceData: {exc}"}))
        sys.exit(1)

    report = generate_report(idata, method=args.method)
    output = json.dumps(report, indent=2, default=json_default)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
