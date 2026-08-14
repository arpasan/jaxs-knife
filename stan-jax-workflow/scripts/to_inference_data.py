"""Convert CmdStanPy, nutpie, or BlackJAX output to ArviZ InferenceData."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from numpy.typing import NDArray

try:
    import arviz as az
except ImportError as exc:  # pragma: no cover
    raise ImportError("arviz is required for to_inference_data") from exc


# ==================================================
# Engine adapters
# ==================================================


def from_cmdstanpy(
    fit: Any,
    *,
    posterior_predictive: Optional[list[str]] = None,
    log_likelihood: Optional[list[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Wrap a CmdStanPy ``CmdStanMCMC`` (or compatible) fit as InferenceData.

    Parameters
    ----------
    fit : Any
        Object accepted by ``arviz.from_cmdstanpy`` as ``posterior``.
    posterior_predictive : list[str], optional
        GQ names to store as PPC. Defaults to ``y_rep`` / ``yrep`` if present.
    log_likelihood : list[str], optional
        GQ names for pointwise log likelihood. Defaults to ``log_lik`` /
        ``log_likelihood`` if present.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Any
        ArviZ InferenceData (or DataTree on ArviZ 1.x).
    """
    if logger is not None:
        logger.info("Converting CmdStanPy fit to InferenceData")
    names: set[str] = set()
    stan_vars = getattr(fit, "stan_variables", None)
    if callable(stan_vars):
        try:
            names = set(stan_vars().keys())
        except Exception:
            names = set()
    pp = posterior_predictive
    if pp is None:
        pp = [n for n in ("y_rep", "yrep") if n in names]
    ll = log_likelihood
    if ll is None:
        ll = [n for n in ("log_lik", "log_likelihood") if n in names]
    return az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=pp or None,
        log_likelihood=ll or None,
    )


def from_nutpie(
    trace: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Pass through a nutpie trace that is already InferenceData-like.

    Parameters
    ----------
    trace : Any
        nutpie ``sample()`` return value, or an object with ``to_inference_data``.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Any
        ArviZ InferenceData.
    """
    if logger is not None:
        logger.info("Normalizing nutpie trace to InferenceData")
    if hasattr(trace, "to_inference_data"):
        return trace.to_inference_data()
    return trace


def from_blackjax(
    posterior: Mapping[str, NDArray[Any]],
    *,
    sample_stats: Optional[Mapping[str, NDArray[Any]]] = None,
    observed_data: Optional[Mapping[str, NDArray[Any]]] = None,
    posterior_predictive: Optional[Mapping[str, NDArray[Any]]] = None,
    log_likelihood: Optional[Mapping[str, NDArray[Any]]] = None,
    coords: Optional[Dict[str, Any]] = None,
    dims: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Build InferenceData from BlackJAX (or any JAX) draw dictionaries.

    Array axes must be ``(chain, draw, ...)``.

    Parameters
    ----------
    posterior : Mapping[str, NDArray]
        Posterior draws.
    sample_stats : Mapping[str, NDArray], optional
        Include ``diverging`` when the sampler exposes it.
    observed_data : Mapping[str, NDArray], optional
        Observed responses for PPC / calibration.
    posterior_predictive : Mapping[str, NDArray], optional
        Generated-quantity / vmap PPC draws.
    log_likelihood : Mapping[str, NDArray], optional
        Pointwise log likelihood for PSIS-LOO.
    coords : dict, optional
        ArviZ coordinates.
    dims : dict, optional
        ArviZ dimension map.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Any
        ArviZ InferenceData.
    """
    if logger is not None:
        logger.info("Building InferenceData from BlackJAX draw dicts")
    payload: Dict[str, Mapping[str, NDArray[Any]]] = {"posterior": dict(posterior)}
    if sample_stats is not None:
        payload["sample_stats"] = dict(sample_stats)
    if observed_data is not None:
        payload["observed_data"] = dict(observed_data)
    if posterior_predictive is not None:
        payload["posterior_predictive"] = dict(posterior_predictive)
    if log_likelihood is not None:
        payload["log_likelihood"] = dict(log_likelihood)
    return az.from_dict(payload, coords=coords, dims=dims)
