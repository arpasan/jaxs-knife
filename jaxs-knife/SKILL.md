---
name: jaxs-knife
description: >
  Bayesian modeling workflow for Stan (CmdStanPy / nutpie[stan]) and
  JAX (log-density + BlackJAX), sharing ArviZ InferenceData. Encodes the
  Gelman–Vehtari sequence, Stan geometry (non-centered, Jacobian, generated
  quantities, diagnose), and JAX-like-Stan log-density patterns that agents skip
  unprompted. Use when writing or diagnosing .stan programs, CmdStan/CmdStanPy
  fits, JAX logdensity_fn / BlackJAX NUTS, hierarchical or funnel geometry,
  partial pooling, measurement error, truncation, censoring, missing data, imperfect assays,
  divergences, R-hat, ESS, PSIS-LOO, ELPD, stacking, prior
  predictive checks, or a Bayesian report.md. Do not use for PyMC-only models,
  causal identification / DAGs /
  DiD / RDD, or simulation-based inference / BayesFlow.
license: MIT
metadata:
  version: "0.1.5"
---

# jaxs-knife

Stan when you can. JAX when you must.

Every analysis follows this sequence. Do not skip criticism.

1. **Formulate** — Generative story first. Driving question first. Bayes is optional if counting suffices. How did a row get into the file, and why is a cell blank? [references/observation.md](references/observation.md).
2. **Specify priors** — [references/priors.md](references/priors.md). Weakly informative. Never `normal(0, 1000)`. Justify every prior. Do not center on a selected slice. Stated quantiles and support are the prior when that is the knowledge.
3. **Implement** — Pick an engine ([references/engines.md](references/engines.md)), then write the model.
4. **Prior predictive** — Before MCMC. If prior-predictive draws are scientifically implausible, fix priors first.
5. **Fake-data recovery** — Simulate from *this* model at known values, fit, and confirm the named estimand is in the **reported** interval. The recovery fit must pass the same diagnose gates. One recovery is not coverage. Recovery cannot bless the observation model. Only then fit the real data.
6. **Inference** — Sample; save draws immediately.
7. **Diagnose** — [references/diagnostics.md](references/diagnostics.md). Refuse to interpret a bad geometry.
8. **Criticize** — [references/model-criticism.md](references/model-criticism.md). PPC in generated quantities (Stan) or `vmap` (JAX). Decision functionals live there too. LOO-PIT for spread and location, not a KDE of the mean. If criticism names a different observation rule, that program is the next fit, not a limitations sentence.
9. **Sensitivity** — [references/sensitivity.md](references/sensitivity.md) when conclusions are decision-relevant.
10. **Compare** — [references/model-comparison.md](references/model-comparison.md) if two or more models. PSIS-LOO over WAIC.
11. **Report** — `<slug>/report.md` from [references/reporting.md](references/reporting.md). Ratings from `scripts/check_diagnostics.py`, not asserted.

## Engine decision

Do not transpile Stan to XLA. Do not treat BridgeStan host-callbacks as a JIT/GPU Stan path.

| If | Then |
|---|---|
| Likelihood is Stan-shaped; hierarchical GLM; need GQ + `diagnose()`; CPU NUTS; CSV reproducibility | Write Stan. Sample with **nutpie[stan]** if BridgeStan works, else **CmdStanPy NUTS**. Convert to ArviZ. |
| GPU density, SVI, custom MCMC, or a density Stan cannot express | Write a JAX log-density **like Stan** (constrain, Jacobian, logp, `vmap` GQ) + **BlackJAX**. NumPyro only if plates help. |
| Same science, two engines | Use Stan MCMC as a reference computation; JAX as the other. Compare in ArviZ. Do not require bit-identical posteriors. |
| Difficult posterior geometry / slow warmup | Pathfinder or nutpie low-rank+diag mass matrix, **then NUTS**. Label Pathfinder/Laplace as approximations. |
| Simulator, no tractable likelihood | Out of scope. Do not substitute ABC. |

## Installation

Prefer conda-forge for compiled bits:

```bash
mamba env create -f environment.yml
# CmdStan 2.39 (once per machine)
python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.39.0')"
```

If nutpie/BridgeStan fails, fall back to `model.sample()` in CmdStanPy. Do not rewrite the model in NumPyro unless plates are required.

Minimum versions: CmdStan 2.39, CmdStanPy 1.3, BridgeStan 2.9 (nutpie may fetch 2.8), nutpie 0.16.11, BlackJAX 1.6.2.

## Stan template

```stan
data {
  int<lower=1> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=0> sigma;  // constrained decl: Jacobian is automatic
}
model {
  mu ~ normal(0, 2.5);      // justify in report.md
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
```

```python
import cmdstanpy
import arviz as az
from pathlib import Path

RANDOM_SEED = sum(map(ord, "analysis-slug-v1"))
model = cmdstanpy.CmdStanModel(stan_file="model.stan")
# Prefer nutpie[stan] when the C++ toolchain works; else:
fit = model.sample(data=data, seed=RANDOM_SEED)
idata = az.from_cmdstanpy(fit)
idata.to_netcdf("slug/inference_data.nc")
```

## JAX template (write it like Stan)

```python
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

jax.config.update("jax_enable_x64", True)  # before the density; record precision in the appendix

def constrain(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    mu, z_sigma = z[0], z[1]
    sigma = jnp.exp(z_sigma)
    log_abs_det = z_sigma  # d(log sigma) / d z_sigma
    return jnp.array([mu, sigma]), log_abs_det

def logdensity_fn(z: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    theta, log_abs_det = constrain(z)
    mu, sigma = theta[0], theta[1]
    # Priors on constrained values (same families as the Stan template).
    # Do not put a density on log(sigma) *and* add J — that double-counts.
    logp = norm.logpdf(mu, 0.0, 2.5) + (-sigma)  # exponential(1) on sigma
    logp = logp + jnp.sum(norm.logpdf(y, mu, sigma))
    return logp + log_abs_det
```

Sample with BlackJAX NUTS. Land draws in ArviZ via `scripts/to_inference_data.py`. PPC with `jax.vmap`, not a numpy rewrite of the likelihood.

## Critical rules

- Prior predictive **before** sampling. Prefer a `prior_only` data flag in the same program over a second, drifted copy.
- Fake-data recovery **before** the real fit: known parameters in, named estimand in the reported interval, recovery fit clears the same gates. One run is not coverage. A recovery from the fitted program does not validate the inclusion rule.
- Rank-normalized split R-hat ≤ 1.01; > 1.05 do not interpret. ESS bulk and tail ≥ 100 × n_chains. Gate ESS on the interval you print.
- Divergences: refuse to interpret. A missing `sample_stats.diverging` flag is unknown, not zero. Reparameterize, then raise `adapt_delta` / `target_accept`.
- E-BFMI and treedepth are first-class (`cmdstan diagnose` when the fit is CmdStan).
- Constrained Stan declarations auto-adjust the Jacobian. Sampling a **transform** of a parameter needs `jacobian +=`.
- Write JAX like Stan: constrain, Jacobian, log density, `vmap` GQ. Enable `jax_enable_x64` before constructing the density.
- Pathfinder / Laplace / ADVI are approximations or NUTS inits unless labeled otherwise.
- PPC *p*-values are not hypothesis tests. PSIS-LOO over WAIC. Pareto-k < 0.5 trust; 0.5–0.7 investigate; ≥ 0.7 K-fold (moment matching needs a callable density, not a CmdStan CSV).
- ΔELPD < 2 × dSE → indistinguishable; prefer the simpler model. Stacking when there is no winner. LOO is not NHST.
- Power-scaling CJS > 0.05 is a flag to document, not a command to loosen priors.
- Posterior **mean** of predictive probabilities, never median.
- Probability language. Never “significant” / p-values. Prefer prior-to-posterior movement over a Bayes factor.
- Compute 50% and 94% HDIs on the scale you print. An HDI is not invariant to reparameterization.
- Save draws before post-processing (`inference_data.nc`).
- Discrete latents and mixtures: [references/mixtures.md](references/mixtures.md). Label switching is not more draws.
- Recording, truncation, censoring, missing cells, measurement error, imperfect assays: [references/observation.md](references/observation.md). The observation model is part of the likelihood.
- Use `scripts/diagnose_model.py` → `calibration_check.py` → `check_diagnostics.py`. Paste ratings into `report.md`.

## Utility scripts

```bash
python scripts/diagnose_model.py --idata <slug>/inference_data.nc --output <slug>/diagnostics.json
python scripts/calibration_check.py --idata <slug>/inference_data.nc --output <slug>/calibration.json
python scripts/check_diagnostics.py --diagnostics <slug>/diagnostics.json --calibration <slug>/calibration.json --output <slug>/check_report.json
```

## Common failure modes

- **HalfCauchy / HalfFlat on hierarchical scales** → funnel. Use `exponential` or `gamma` and non-centered when groups are weakly identified.
- **Missing Jacobian** on a sampled transform → wrong posterior. Prefer `<lower=0>` decls.
- **PPC in Python** after a Stan fit → you reimplemented the likelihood. Use `generated quantities`.
- **nutpie has no GQ** → run CmdStan `generate_quantities` or compute PPC/log_lik in JAX/`vmap` after the fact; still convert to ArviZ.
- **Pathfinder draws reported as MCMC** → mislabeled approximation.
- **NumPyro 8-schools HalfCauchy** → override; it is a known funnel prior.
- **BridgeStan + BlackJAX callbacks** → Stan grads are C++, not `jit`/`gpu` of the Stan density.
- **`np.median` on predictive probabilities** → not the posterior predictive.
- **Off-by-one group index** → Stan is 1-based; reuse one level map for any prediction data.
- **Selected slice, complete-data likelihood** → interval is for the retained rows, not the process. Write the inclusion rule first.
- **`y ~ x_obs` when `x` has a reported SE** → attenuated slope, over-confident interval. The regressor is the unmarked true value.
- **Assay calls treated as truth** → prevalence pinned to the false-positive floor. The stated error rate belongs in the likelihood.
- **`dropna` on incomplete rows** → an inclusion rule. A blank is not automatically a latent draw from the untruncated density. If recorded values stop at a ceiling (or start at a floor) while a complete-data PPC walks past that edge, the blank rows get a tail probability (`*_lccdf` / `*_lcdf`), then you **refit**. Imputing blanks from `y | x` under MAR given `x` is the same posterior as deleting the rows. A logistic missingness grid is not a censored likelihood. Naming that next model in limitations is not a fit.
- **HDI of `log(theta)`** → not the log of the HDI of `theta`. Compute the interval on the scientific scale.
- **New group scored as `mu` or a fitted `theta[j]`** → draw from the hyperprior in generated quantities.
- **i.i.d. Poisson, then NB, when rows have unequal exposure** → offset first; overdispersion is a later repair.

## When things go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Init fail / NaN logp | Data, bounds, or precision | Fix decls / dtypes / init; `jax_enable_x64`. Not `adapt_delta` |
| Missing `diverging` | JAX stats not stored | Store `is_divergent` as `diverging`. Missing ≠ 0 |
| Divergences | Funnel / missing Jacobian / bad scales | Non-centered; constrained decl; then `adapt_delta` 0.95–0.99 |
| R-hat > 1.01 | Poor mixing / multimodality | More draws; Pathfinder init; pair plots |
| Label-switching R-hat | Unordered mixture | `ordered[K]` / `log_mix`; [mixtures.md](references/mixtures.md) |
| Low ESS | Autocorrelation | Reparameterize; QR if `X` is collinear |
| Treedepth saturation | Difficult geometry | Reparameterize before raising `max_treedepth` |
| Prior pred. nonsense | Bad priors | Tighten; never `normal(0, 1000)` |
| PPC looks fine; slope too small | Predictor is an instrument reading | Latent true `x`; measurement likelihood on the printed value |
| PPC looks fine; mean off a cutoff | Truncation / selection | Retention probability in the likelihood, not a complete-data density |
| Recorded max (min) far below (above) almost all `y_rep` extrema | Reporting limit / censoring | Tail probability on the blank or piled-up rows; **refit**. Do not stop at MAR imputation or a limitations sentence |
| Prevalence near the assay floor | Labels are calls | Put the stated error rates in the observation model |
| PPC misses data | Misspecification | One repair from the observed misfit; then refit |
| Pareto-k ≥ 0.7 | Influential points | Inspect; Student-t; K-fold |
| nutpie compile fail | Toolchain / BridgeStan | CmdStanPy `sample()` fallback |

## Sources

- Gelman, Vehtari, Simpson, Margossian, Carpenter, Yao, Kennedy, Gabry, Bürkner, and Modrák (2020), [Bayesian Workflow](https://arxiv.org/abs/2011.01808).
- Vehtari, Gelman, Simpson, Carpenter, and Bürkner (2021), rank-normalized split R-hat and ESS.
- Vehtari, Gelman, and Gabry (2017), PSIS-LOO and WAIC.
- Carpenter (2025), [It’s a JAX, JAX, JAX, JAX World](https://statmodeling.stat.columbia.edu/2025/10/03/its-a-jax-jax-jax-jax-world/) — write the density in JAX; do not transpile Stan to XLA.
