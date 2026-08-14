"""Live NUTS smoke: CmdStanPy and BlackJAX on the same mini-normal data."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "stan-jax-workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from check_diagnostics import check_diagnostics, suggest_next_steps
from diagnose_model import generate_report
from json_util import json_default
from to_inference_data import from_blackjax, from_cmdstanpy

ARTIFACTS = ROOT / "evals" / "smoke" / "artifacts"
WORK = ROOT / "evals" / "smoke" / "_work"

SEED = sum(map(ord, "mini-normal-live-v1"))
N_OBS = 40
N_CHAINS = 4
N_WARMUP = 400
N_DRAWS = 400

STAN_CODE = """
data {
  int<lower=1> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=1e-8> sigma;
}
model {
  mu ~ normal(0, 2.5);
  sigma ~ exponential(1);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
  }
}
"""


# ==================================================
# Shared data and toolchain
# ==================================================


def simulate_data(seed: int = SEED) -> NDArray[np.float64]:
    """Draw the shared observation vector.

    Parameters
    ----------
    seed : int
        RNG seed.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(N_OBS,)``.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.3, 1.0, size=N_OBS).astype(np.float64)


def ensure_cmdstan(logger: logging.Logger | None = None) -> Path:
    """Return the CmdStan path, installing 2.39.0 if it is missing.

    Parameters
    ----------
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Path
        CmdStan installation directory.
    """
    import cmdstanpy

    try:
        path = Path(cmdstanpy.cmdstan_path())
        if logger is not None:
            logger.info("CmdStan at %s", path)
        return path
    except Exception:
        if logger is not None:
            logger.info("CmdStan missing; installing 2.39.0")
        cmdstanpy.install_cmdstan(version="2.39.0")
        return Path(cmdstanpy.cmdstan_path())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )


def _bundle(idata: Any) -> Dict[str, Any]:
    diagnostics = generate_report(idata)
    report = check_diagnostics(diagnostics=diagnostics)
    report["next_steps"] = suggest_next_steps(report)
    report["diagnostics"] = diagnostics
    return report


# ==================================================
# Stan
# ==================================================


def fit_stan(
    y: NDArray[np.float64],
    *,
    logger: logging.Logger | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Compile and sample the mini-normal model with CmdStanPy NUTS.

    Parameters
    ----------
    y : NDArray[np.float64]
        Observations.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        InferenceData and the diagnostic report.
    """
    import cmdstanpy

    ensure_cmdstan(logger=logger)
    work = WORK / "stan"
    work.mkdir(parents=True, exist_ok=True)
    stan_path = work / "mini_normal.stan"
    stan_path.write_text(STAN_CODE, encoding="utf-8")
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_path))
    fit = model.sample(
        data={"N": int(y.size), "y": y.tolist()},
        chains=N_CHAINS,
        iter_warmup=N_WARMUP,
        iter_sampling=N_DRAWS,
        seed=SEED,
        show_progress=False,
    )
    idata = from_cmdstanpy(fit, observed_data={"y": y}, logger=logger)
    return idata, _bundle(idata)


# ==================================================
# JAX / BlackJAX
# ==================================================


def fit_blackjax(
    y: NDArray[np.float64],
    *,
    logger: logging.Logger | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    """BlackJAX NUTS on a constrain + Jacobian log-density.

    Parameters
    ----------
    y : NDArray[np.float64]
        Observations.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        InferenceData and the diagnostic report.
    """
    import blackjax
    import jax
    import jax.numpy as jnp
    from jax.scipy.stats import norm

    y_jax = jnp.asarray(y)

    def logdensity_fn(z: Any) -> Any:
        mu = z[0]
        log_sigma = z[1]
        sigma = jnp.exp(log_sigma)
        log_abs_det = log_sigma
        logp = norm.logpdf(mu, 0.0, 2.5)
        logp = logp - sigma
        logp = logp + jnp.sum(norm.logpdf(y_jax, mu, sigma))
        return logp + log_abs_det

    def run_chain(rng_key: Any, init: Any) -> Tuple[Any, Any]:
        warmup_key, sample_key = jax.random.split(rng_key)
        adapt = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
        (state, params), _ = adapt.run(warmup_key, init, num_steps=N_WARMUP)
        step = blackjax.nuts(logdensity_fn, **params).step

        def one_step(carry: Any, key: Any) -> Tuple[Any, Tuple[Any, Any]]:
            new_state, info = step(key, carry)
            return new_state, (new_state.position, info.is_divergent)

        keys = jax.random.split(sample_key, N_DRAWS)
        _, (positions, diverging) = jax.lax.scan(one_step, state, keys)
        return positions, diverging

    rng = np.random.default_rng(SEED + 1)
    inits = jnp.asarray(rng.normal(0.0, 0.3, size=(N_CHAINS, 2)))
    keys = jax.random.split(jax.random.PRNGKey(SEED), N_CHAINS)
    positions, diverging = jax.vmap(run_chain)(keys, inits)
    # positions: (chain, draw, 2)
    mu = np.asarray(positions[:, :, 0])
    sigma = np.exp(np.asarray(positions[:, :, 1]))
    div = np.asarray(diverging, dtype=bool)

    ppc_key = jax.random.PRNGKey(SEED + 2)
    noise = jax.random.normal(ppc_key, shape=(N_CHAINS, N_DRAWS, y.size))
    y_rep = np.asarray(mu[:, :, None] + sigma[:, :, None] * noise)
    log_lik = np.asarray(
        jax.vmap(
            jax.vmap(lambda m, s: norm.logpdf(y_jax, m, s)),
            in_axes=(0, 0),
        )(jnp.asarray(mu), jnp.asarray(sigma))
    )

    if logger is not None:
        logger.info("BlackJAX NUTS finished; divergences=%s", int(div.sum()))

    idata = from_blackjax(
        {"mu": mu, "sigma": sigma},
        sample_stats={"diverging": div},
        observed_data={"y": y},
        posterior_predictive={"y": y_rep},
        log_likelihood={"y": log_lik},
        logger=logger,
    )
    return idata, _bundle(idata)


# ==================================================
# Persist
# ==================================================


def write_run(
    engine: str,
    idata: Any,
    report: Dict[str, Any],
) -> Path:
    """Write InferenceData, JSON, and return the engine artifact directory.

    Parameters
    ----------
    engine : str
        ``stan`` or ``jax``.
    idata : Any
        ArviZ object.
    report : dict
        Diagnostic report.

    Returns
    -------
    Path
        Artifact directory.
    """
    out = ARTIFACTS / engine
    out.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(out / "inference_data.nc")
    _write_json(out / "diagnostics.json", report["diagnostics"])
    slim = {k: v for k, v in report.items() if k != "diagnostics"}
    _write_json(out / "check_report.json", slim)
    return out


def run_both(*, logger: logging.Logger | None = None) -> Dict[str, Dict[str, Any]]:
    """Fit both engines on the same ``y`` and write artifacts.

    Parameters
    ----------
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        ``{"stan": report, "jax": report}``.
    """
    y = simulate_data()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS / "y.npy", y)

    stan_idata, stan_report = fit_stan(y, logger=logger)
    write_run("stan", stan_idata, stan_report)

    jax_idata, jax_report = fit_blackjax(y, logger=logger)
    write_run("jax", jax_idata, jax_report)

    from gallery import write_gallery

    write_gallery(stan_idata, jax_idata, stan_report, jax_report)
    return {"stan": stan_report, "jax": jax_report}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    reports = run_both(logger=logging.getLogger("live_nuts"))
    summary = {
        engine: {
            "convergence": reports[engine]["convergence"]["rating"],
            "divergences": reports[engine]["diagnostics"]["convergence"]["divergences"],
        }
        for engine in reports
    }
    print(json.dumps(summary, indent=2, default=json_default))
    print(f"Gallery: {ARTIFACTS / 'index.html'}")


if __name__ == "__main__":
    main()
