---
name: stan-jax-workflow
description: >
  Opinionated Bayesian modeling workflow for Stan (CmdStanPy / nutpie[stan]) and
  JAX (log-density + BlackJAX), sharing ArviZ InferenceData. Encodes the
  Gelman–Vehtari sequence, Stan geometry (non-centered, Jacobian, generated
  quantities, diagnose), and JAX-like-Stan log-density patterns that agents skip
  unprompted. Use when writing or diagnosing .stan programs, CmdStan/CmdStanPy
  fits, JAX logdensity_fn / BlackJAX NUTS, hierarchical or funnel geometry,
  partial pooling, divergences, R-hat, ESS, PSIS-LOO, ELPD, stacking, prior
  predictive checks, or a Bayesian report.md. Do not use for PyMC-only models,
  causal identification / DAGs /
  DiD / RDD, or simulation-based inference / BayesFlow.
license: MIT
metadata:
  version: "0.1.0"
---

# Stan / JAX Bayesian Workflow

Every analysis follows this sequence. Do not skip criticism.

1. **Formulate** — Generative story first. Driving question first. Bayes is optional if counting suffices.
2. **Specify priors** — [references/priors.md](references/priors.md). Weakly informative. Never `normal(0, 1000)`. Justify every prior.
3. **Implement** — Pick an engine ([references/engines.md](references/engines.md)), then write the model.
4. **Prior predictive** — Before MCMC. If draws are nonsense, fix priors first.
5. **Inference** — Sample; save draws immediately.
6. **Diagnose** — [references/diagnostics.md](references/diagnostics.md). Refuse to interpret a bad geometry.
7. **Criticize** — [references/model-criticism.md](references/model-criticism.md). PPC in generated quantities (Stan) or `vmap` (JAX).
8. **Sensitivity** — [references/sensitivity.md](references/sensitivity.md) when conclusions are decision-relevant.
9. **Compare** — [references/model-comparison.md](references/model-comparison.md) if two or more models. PSIS-LOO over WAIC.
10. **Report** — `<slug>/report.md` from [references/reporting.md](references/reporting.md). Ratings from `scripts/check_diagnostics.py`, not vibes.

## Engine decision

Stan stays. JAX is a peer, not a replacement. Do not transpile Stan to XLA. Do not treat BridgeStan host-callbacks as a JIT/GPU Stan path.

| If | Then |
|---|---|
| Likelihood is Stan-shaped; hierarchical GLM; need GQ + `diagnose()`; CPU NUTS; CSV reproducibility | Write Stan. Sample with **nutpie[stan]** if BridgeStan works, else **CmdStanPy NUTS**. Convert to ArviZ. |
| GPU density, SVI, custom MCMC, or a density Stan cannot express | Write a JAX log-density **like Stan** (constrain, Jacobian, logp, `vmap` GQ) + **BlackJAX**. NumPyro only if plates help. |
| Same science, two engines | Stan MCMC as gold standard; JAX as the other. Compare in ArviZ. Do not require bit-identical posteriors. |
| Horrible geometry / slow warmup | Pathfinder or nutpie low-rank+diag mass matrix, **then NUTS**. Label Pathfinder/Laplace as approximations. |
| Simulator, no tractable likelihood | Out of scope. Do not fake ABC. Do not clone BayesFlow. |

## Installation

Prefer conda-forge for compiled bits:

```bash
mamba env create -f environment.yml
# CmdStan 2.39 (once per machine)
python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.39.0')"
```

If nutpie/BridgeStan fails, fall back to `model.sample()` in CmdStanPy. Do not rewrite the model in NumPyro for fashion.

Pinned: CmdStan 2.39, CmdStanPy 1.3, BridgeStan 2.9 (nutpie may fetch 2.8), nutpie 0.16.11, BlackJAX 1.6.2.

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

def constrain(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    mu, z_sigma = z[0], z[1]
    sigma = jnp.exp(z_sigma)
    log_abs_det = z_sigma  # d/dz log sigma
    return jnp.array([mu, sigma]), log_abs_det

def logdensity_fn(z: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    theta, log_abs_det = constrain(z)
    mu, sigma = theta[0], theta[1]
    logp = norm.logpdf(mu, 0.0, 2.5) + norm.logpdf(jnp.log(sigma), 0.0, 1.0)
    logp = logp + jnp.sum(norm.logpdf(y, mu, sigma))
    return logp + log_abs_det
```

Sample with BlackJAX NUTS. Land draws in ArviZ via `scripts/to_inference_data.py`. PPC with `jax.vmap`, not a numpy rewrite of the likelihood.

## Critical rules

- Prior predictive **before** sampling.
- Rank-normalized split R-hat ≤ 1.01; > 1.05 do not interpret. ESS bulk and tail ≥ 100 × n_chains.
- Divergences: refuse to interpret. Reparameterize, then raise `adapt_delta` / `target_accept`.
- E-BFMI and treedepth are first-class (`cmdstan diagnose` when the fit is CmdStan).
- Constrained Stan declarations auto-adjust the Jacobian. Sampling a **transform** of a parameter needs `jacobian +=`.
- Write JAX like Stan: constrain, Jacobian, log density, `vmap` GQ.
- Pathfinder / Laplace / ADVI are approximations or NUTS inits unless labeled otherwise.
- PPC *p*-values are not hypothesis tests. PSIS-LOO over WAIC. Pareto-k < 0.5 trust; 0.5–0.7 investigate; ≥ 0.7 K-fold or moment matching.
- ΔELPD < 2 × dSE → indistinguishable; prefer the simpler model. Stacking when there is no winner. LOO is not NHST.
- Power-scaling CJS > 0.05 is a flag to document, not a command to loosen priors.
- Intervals: report **50% typical + 94% range**. No width is magic. 80% HDI is notebook ink, not the scientific default. HDI, not bare ETI.
- Posterior **mean** of predictive probabilities, never median.
- Probability language. Never “significant” / p-values.
- Save draws before post-processing (`inference_data.nc`).
- Identifiability: pair plots near ±1 → merge components. Discrete latents: marginalize.
- Use `scripts/diagnose_model.py` → `calibration_check.py` → `check_diagnostics.py`. Paste ratings into `report.md`.

## Utility scripts

```bash
python scripts/diagnose_model.py --idata <slug>/inference_data.nc --output <slug>/diagnostics.json
python scripts/calibration_check.py --idata <slug>/inference_data.nc --output <slug>/calibration.json
python scripts/check_diagnostics.py --diagnostics <slug>/diagnostics.json --calibration <slug>/calibration.json --output <slug>/check_report.json
```

## Common gotchas

- **HalfCauchy / HalfFlat on hierarchical scales** → funnel. Use `exponential` or `gamma` and non-centered when groups are weakly identified.
- **Missing Jacobian** on a sampled transform → wrong posterior. Prefer `<lower=0>` decls.
- **PPC in Python** after a Stan fit → you reimplemented the likelihood. Use `generated quantities`.
- **nutpie has no GQ** → run CmdStan `generate_quantities` or compute PPC/log_lik in JAX/`vmap` after the fact; still convert to ArviZ.
- **Pathfinder draws reported as MCMC** → mislabeled approximation.
- **NumPyro 8-schools HalfCauchy** → override; it is a known funnel prior.
- **BridgeStan + BlackJAX callbacks** → Stan grads are C++, not `jit`/`gpu` of the Stan density.
- **`np.median` on predictive probabilities** → not the posterior predictive.

## When things go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Divergences | Funnel / missing Jacobian / bad scales | Non-centered; constrained decl; then `adapt_delta` 0.95–0.99 |
| R-hat > 1.01 | Poor mixing / multimodality | More draws; Pathfinder init; pair plots |
| Low ESS | Autocorrelation | Reparameterize; QR if `X` is collinear |
| Treedepth saturation | Difficult geometry | Reparameterize before raising `max_treedepth` |
| Prior pred. nonsense | Bad priors | Tighten; never `normal(0, 1000)` |
| PPC misses data | Misspecification | Heavier tails, overdispersion, hierarchy |
| Pareto-k ≥ 0.7 | Influential points | Inspect; Student-t; K-fold |
| nutpie compile fail | Toolchain / BridgeStan | CmdStanPy `sample()` fallback |
