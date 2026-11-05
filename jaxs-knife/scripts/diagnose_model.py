"""Convergence, LOO, and PPC presence checks on ArviZ InferenceData.

No PyMC dependency. LOO requires a ``log_likelihood`` group already on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    azs = None
    HAS_DIAGNOSE = False


# ==================================================
# Inference method and ESS gates
# ==================================================

MIN_NUTS_CHAINS = 4
_CHAIN_FLOOR_MARKER = "n_chains<4"
_APPROX_METHODS = frozenset({"pathfinder", "vi", "laplace"})
_ESS_SKIP = ("y_rep", "yrep", "log_lik", "log_likelihood", "lp__")


def _canonical_method(label: str) -> str:
    """Map a free-text method label onto nuts / pathfinder / vi / laplace."""
    key = " ".join(label.strip().lower().replace("-", " ").replace("_", " ").split())
    tokens = set(key.split())
    if "pathfinder" in tokens:
        return "pathfinder"
    if "laplace" in tokens:
        return "laplace"
    if tokens & {"vi", "advi", "variational", "meanfield"}:
        return "vi"
    if tokens & {"nuts", "hmc", "mcmc", "blackjax", "nutpie", "cmdstan", "cmdstanpy", "stan"}:
        return "nuts"
    return key


def _attr_get(obj: Any, key: str) -> Optional[str]:
    """Read one attrs entry. Missing object or key returns None."""
    if obj is None:
        return None
    attrs = getattr(obj, "attrs", None)
    if attrs is None:
        return None
    try:
        value = attrs.get(key)
    except Exception:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_inference_method(idata: Any, inference_method: Optional[str] = None) -> str:
    """Resolve the sampler label used by the ESS chain floor.

    Explicit argument wins, then ``idata.attrs['inference_method']``, then
    ``posterior.attrs``. A missing label is NUTS, so an unlabeled one-chain
    fit fails the chain floor. Pathfinder, VI, and Laplace must be labeled
    to skip that floor.

    Parameters
    ----------
    idata : Any
        ArviZ InferenceData or DataTree.
    inference_method : str, optional
        Caller-supplied label (CLI ``--method`` or library argument).

    Returns
    -------
    str
        ``nuts``, ``pathfinder``, ``vi``, ``laplace``, or a normalized token.
    """
    if inference_method is not None and str(inference_method).strip():
        return _canonical_method(str(inference_method))
    for obj in (idata, getattr(idata, "posterior", None)):
        found = _attr_get(obj, "inference_method")
        if found:
            return _canonical_method(found)
    return "nuts"


def _n_chains(idata: Any) -> int:
    """Chain count on the posterior. Missing chain dim counts as 1."""
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return 1
    sizes = getattr(posterior, "sizes", None)
    if sizes is not None and "chain" in sizes:
        return int(sizes["chain"])
    chain = getattr(posterior, "chain", None)
    if chain is None:
        return 1
    return int(getattr(chain, "size", 1) or 1)


def _filtered_names(params: Any, skip: Tuple[str, ...]) -> List[str]:
    """Stringify diagnostic names, dropping generated quantities."""
    if not params:
        return []
    kept: List[str] = []
    for param in params:
        text = str(param)
        low = text.lower()
        if low.startswith(skip) or low in {"lp__", "energy__"}:
            continue
        kept.append(text)
    return kept


def _apply_chain_floor(params: List[str], *, n_chains: int, method: str) -> List[str]:
    """Fail the ESS gate for NUTS when fewer than 4 chains are present.

    Pathfinder, VI, and Laplace are exempt. The marker is prepended so the
    numeric ESS failures stay visible.
    """
    if method in _APPROX_METHODS or n_chains >= MIN_NUTS_CHAINS:
        return params
    if _CHAIN_FLOOR_MARKER in params:
        return params
    return [_CHAIN_FLOOR_MARKER, *params]


def _ess_names_below(ds: Any, *, threshold: float, skip: Tuple[str, ...]) -> List[str]:
    """Parameter names whose ESS array falls below ``threshold``."""
    data_vars = getattr(ds, "data_vars", None)
    if data_vars is None:
        return []
    bad: List[str] = []
    for name in data_vars:
        text = str(name)
        low = text.lower()
        if low.startswith(skip) or low in {"lp__", "energy__"}:
            continue
        arr = np.asarray(ds[name].values, dtype=float)
        if arr.size == 0 or (not np.all(np.isfinite(arr))) or bool(np.any(arr < threshold)):
            bad.append(text)
    return bad


def _ess_tail_bad_params(
    idata: Any,
    diag: Dict[str, Any],
    *,
    n_chains: int,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Tail-ESS failures. Never a copy of the bulk list.

    Prefer an ``ess_tail`` block on the diagnose payload. Otherwise compute
    tail ESS (``method='tail'``) against ``100 * n_chains``.
    """
    tail_block = diag.get("ess_tail")
    if isinstance(tail_block, dict) and "bad_params" in tail_block:
        return _filtered_names(tail_block.get("bad_params", []), _ESS_SKIP)
    ess_block = diag.get("ess")
    if isinstance(ess_block, dict) and isinstance(ess_block.get("ess_tail"), dict):
        nested = ess_block["ess_tail"]
        if "bad_params" in nested:
            return _filtered_names(nested.get("bad_params", []), _ESS_SKIP)
    names = _parameter_names(idata)
    try:
        kwargs: Dict[str, Any] = {"method": "tail"}
        if names:
            kwargs["var_names"] = names
        ds = az.ess(idata, **kwargs)
    except Exception:
        if logger is not None:
            logger.exception("ess_tail diagnostic failed")
        return ["ess_tail_unavailable"]
    return _ess_names_below(ds, threshold=float(100 * n_chains), skip=_ESS_SKIP)


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
    inference_method: Optional[str] = None,
) -> Dict[str, Any]:
    """R-hat, ESS, divergences, and (when available) E-BFMI / treedepth.

    NUTS with fewer than 4 chains fails the ESS gate even when the numeric
    ESS clears ``100 * n_chains``. Pathfinder, VI, and Laplace are exempt.

    Parameters
    ----------
    idata : Any
        ArviZ InferenceData or DataTree.
    logger : logging.Logger, optional
        Injected logger.
    inference_method : str, optional
        Sampler label. See ``resolve_inference_method``.

    Returns
    -------
    Dict[str, Any]
        Serializable convergence block for ``check_diagnostics``.
    """
    method_label = resolve_inference_method(idata, inference_method)
    n_chains = _n_chains(idata)
    divergences = _divergence_block(idata)
    if logger is not None and method_label not in _APPROX_METHODS and n_chains < MIN_NUTS_CHAINS:
        logger.info(
            "ESS chain floor failed: inference_method=%s n_chains=%s",
            method_label,
            n_chains,
        )
    if HAS_DIAGNOSE and azs is not None:
        _has_errors, diag = azs.diagnose(
            idata, return_diagnostics=True, show_diagnostics=False
        )
        diag = diag or {}
        rhat_block = diag.get("rhat") or {}
        ess_block = diag.get("ess") or {}
        rhat_bad = _filtered_names(rhat_block.get("bad_params", []), _ESS_SKIP)
        ess_bulk_bad = _filtered_names(ess_block.get("bad_params", []), _ESS_SKIP)
        ess_tail_bad = _ess_tail_bad_params(
            idata, diag, n_chains=n_chains, logger=logger
        )
        ess_bulk_bad = _apply_chain_floor(
            ess_bulk_bad, n_chains=n_chains, method=method_label
        )
        ess_tail_bad = _apply_chain_floor(
            ess_tail_bad, n_chains=n_chains, method=method_label
        )
        td = diag.get("treedepth") or {}
        n_td = int(td.get("n_max", 0))
        bfmi = diag.get("bfmi") or {}
        failed_chains = [int(c) for c in bfmi.get("failed_chains", [])]
        all_ok = (
            len(rhat_bad) == 0
            and len(ess_bulk_bad) == 0
            and len(ess_tail_bad) == 0
            and bool(divergences.get("ok"))
            and n_td == 0
            and len(failed_chains) == 0
        )
        if logger is not None:
            logger.info(
                "Convergence via arviz_stats.diagnose; all_ok=%s method=%s n_chains=%s",
                all_ok,
                method_label,
                n_chains,
            )
        return {
            "all_ok": all_ok,
            "method": "arviz_stats.diagnose",
            "inference_method": method_label,
            "n_chains": n_chains,
            "rhat": {
                "ok": len(rhat_bad) == 0,
                "max": _max_from_dataset(rhat_block.get("rhat_values")),
                "problematic_params": rhat_bad,
            },
            "ess_bulk": {"ok": len(ess_bulk_bad) == 0, "problematic_params": ess_bulk_bad},
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
        threshold = 100 * n_chains
        rhat = summary["r_hat"]
        rhat_ok = bool(rhat.notna().all() and (rhat <= 1.01).all())
        bulk_params = [
            str(name)
            for name in summary[
                summary["ess_bulk"].isna() | (summary["ess_bulk"] < threshold)
            ].index
        ]
        tail_params = [
            str(name)
            for name in summary[
                summary["ess_tail"].isna() | (summary["ess_tail"] < threshold)
            ].index
        ]
        bulk_params = _apply_chain_floor(
            bulk_params, n_chains=n_chains, method=method_label
        )
        tail_params = _apply_chain_floor(
            tail_params, n_chains=n_chains, method=method_label
        )
        ess_bulk_ok = bool((summary["ess_bulk"] >= threshold).all()) and (
            _CHAIN_FLOOR_MARKER not in bulk_params
        )
        ess_tail_ok = bool((summary["ess_tail"] >= threshold).all()) and (
            _CHAIN_FLOOR_MARKER not in tail_params
        )

        results: Dict[str, Any] = {
            "rhat": {
                "max": float(rhat.max()) if rhat.notna().any() else None,
                "ok": rhat_ok,
                "problematic_params": list(summary[(rhat > 1.01) | rhat.isna()].index),
            },
            "ess_bulk": {
                "min": int(summary["ess_bulk"].min()),
                "ok": ess_bulk_ok,
                "problematic_params": bulk_params,
            },
            "ess_tail": {
                "min": int(summary["ess_tail"].min()),
                "ok": ess_tail_ok,
                "problematic_params": tail_params,
            },
            "divergences": divergences,
            "method": "manual",
            "inference_method": method_label,
            "n_chains": n_chains,
        }
        results["all_ok"] = all(
            results[k]["ok"] for k in ("rhat", "ess_bulk", "ess_tail", "divergences")
        )
        if logger is not None:
            logger.info(
                "Convergence via az.summary; all_ok=%s method=%s n_chains=%s",
                results["all_ok"],
                method_label,
                n_chains,
            )
        return results
    except Exception as exc:
        if logger is not None:
            logger.exception("az.summary fallback failed")
        return {
            "all_ok": False,
            "method": "manual",
            "inference_method": method_label,
            "n_chains": n_chains,
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
    inference_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Full diagnostics report from InferenceData.

    Parameters
    ----------
    idata : Any
        ArviZ InferenceData or DataTree.
    logger : logging.Logger, optional
        Injected logger.
    inference_method : str, optional
        Sampler label forwarded to ``check_convergence``.

    Returns
    -------
    Dict[str, Any]
        ``convergence``, ``loo``, ``posterior_predictive``, ``overall``.
    """
    report: Dict[str, Any] = {
        "convergence": check_convergence(
            idata, logger=logger, inference_method=inference_method
        ),
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
            "Inference method (nuts, pathfinder, vi, laplace). "
            "Defaults to the inference_method attr on InferenceData, else nuts."
        ),
    )
    args = parser.parse_args()

    try:
        idata = az.from_netcdf(args.idata)
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": f"Could not load InferenceData: {exc}"}))
        sys.exit(1)

    report = generate_report(idata, inference_method=args.method)
    output = json.dumps(report, indent=2, default=json_default)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
