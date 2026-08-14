"""Local smoke: synthetic JAX-dict path through diagnose → check_diagnostics.

Optional Stan path runs only if cmdstanpy and a CmdStan install are present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stan-jax-workflow" / "scripts"))

from check_diagnostics import check_diagnostics, suggest_next_steps
from diagnose_model import generate_report
from to_inference_data import from_blackjax


def jax_dict_smoke() -> dict:
    rng = np.random.default_rng(sum(map(ord, "smoke-jax-v1")))
    n_chains, n_draws, n_obs = 4, 200, 10
    mu = rng.normal(0.0, 0.2, size=(n_chains, n_draws))
    sigma = np.exp(rng.normal(0.0, 0.05, size=(n_chains, n_draws)))
    y = rng.normal(0.0, 1.0, size=(n_obs,))
    y_rep = rng.normal(0.0, 1.0, size=(n_chains, n_draws, n_obs))
    idata = from_blackjax(
        {"mu": mu, "sigma": sigma},
        sample_stats={"diverging": np.zeros((n_chains, n_draws), dtype=bool)},
        observed_data={"y": y},
        posterior_predictive={"y": y_rep},
    )
    diagnostics = generate_report(idata)
    report = check_diagnostics(diagnostics=diagnostics)
    report["next_steps"] = suggest_next_steps(report)
    report["engine"] = "jax-dict"
    return report


def stan_smoke() -> dict:
    import cmdstanpy

    stan = """
    data { int<lower=1> N; vector[N] y; }
    parameters { real mu; real<lower=0> sigma; }
    model {
      mu ~ normal(0, 2.5);
      sigma ~ exponential(1);
      y ~ normal(mu, sigma);
    }
    generated quantities {
      vector[N] y_rep;
      for (n in 1:N) y_rep[n] = normal_rng(mu, sigma);
    }
    """
    work = Path("/tmp/standoff-bayes-smoke")
    work.mkdir(parents=True, exist_ok=True)
    stan_path = work / "mini_normal.stan"
    stan_path.write_text(stan, encoding="utf-8")
    rng = np.random.default_rng(sum(map(ord, "smoke-stan-v1")))
    y = rng.normal(0.0, 1.0, size=20)
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_path))
    fit = model.sample(data={"N": 20, "y": y.tolist()}, chains=2, iter_sampling=200, seed=1)
    from to_inference_data import from_cmdstanpy

    idata = from_cmdstanpy(fit)
    diagnostics = generate_report(idata)
    report = check_diagnostics(diagnostics=diagnostics)
    report["next_steps"] = suggest_next_steps(report)
    report["engine"] = "cmdstanpy"
    return report


def main() -> None:
    out = {"jax_dict": jax_dict_smoke()}
    try:
        import cmdstanpy

        cmdstanpy.cmdstan_path()
        out["stan"] = stan_smoke()
    except Exception as exc:
        out["stan"] = {"skipped": str(exc)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
