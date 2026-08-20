# Engines: Stan and JAX

Same scientific workflow. Different log-density languages. ArviZ is the meeting point.

## When to write Stan

Hierarchical GLMs, need generated quantities and `diagnose()`, CPU NUTS, CSV reproducibility. Sample with nutpie[stan] if BridgeStan compiles; else CmdStanPy `sample()`.

Stan constrained declarations (`<lower=0>`) add the Jacobian automatically.

**Change of variables:** if you **sample** a transformed parameter and put a density on the transformed value, add `jacobian +=`. If the transform is only a derived quantity, do not.

PPC and pointwise `log_lik` belong in `generated quantities`, not a numpy loop after the fact.

`cmdstan diagnose` reports energy and treedepth. Use it.

QR, non-centered, and ordered/simplex constraints: [Stan User’s Guide — reparameterization](https://mc-stan.org/docs/stan-users-guide/reparameterization.html).

nutpie does **not** run generated quantities. After nutpie NUTS, either run CmdStan `generate_quantities` on the same draws or compute PPC/`log_lik` in a second pass. Still land in ArviZ.

## When to write JAX

GPU density, SVI, a density Stan cannot express, or a new sampler (BlackJAX). Write the log-density **like Stan** (constrain, Jacobian, logp, `vmap` GQ):

1. Unconstrained parameters in, `constrain` out.
2. Add `log|det J|` of the constraint.
3. Prior + likelihood on **constrained** values (same families as the Stan model).
   A prior already written on the unconstrained coordinate must **not** also
   add `J` — that double-counts the transform.
4. `vmap` the observation model for GQ / PPC, including decision functionals.

Enable `jax.config.update("jax_enable_x64", True)` before constructing the density. JAX defaults to float32; that degrades NUTS adaptation and tail ESS. Record the precision in the report appendix.

Before trusting a hand-written `logdensity_fn`, check finite-difference vs. autodiff gradients and log-density agreement up to a constant against an independent reference (Stan, a tested bijector, or a second implementation). Prefer a tested bijector to a hand-rolled transform.

BlackJAX is a sampler library, not a PPL. You owe it a `logdensity_fn`.

NumPyro is optional when plates help. Override the classic 8-schools HalfCauchy example.

## What not to do

- Transpile Stan → XLA.
- Treat BridgeStan + JAX callbacks as a `jit`/`gpu` Stan path (grads stay C++).
- Report Pathfinder / Laplace as the posterior unless the user asked for an approximation.
- Rewrite a Stan-shaped GLM in NumPyro without a reason the Stan language cannot express.
- Likelihood-free / simulation-based inference (no tractable likelihood).

## Data contract (Stan side)

Stan arrays are 1-based. Build the group map **once** and reuse it for any prediction data.

```stan
array[N] int<lower=1, upper=J> gg;
```

`pandas` category codes need `+ 1`. `int` slots reject float arrays — counts must be integer dtypes. An off-by-one group index can pass every convergence gate and still be the wrong pooling structure.

Pass feature and group names into ArviZ (`coords` / `dims` on `from_cmdstanpy` / `from_blackjax`) so reports are not `theta[0]`.

## Conversion

`scripts/to_inference_data.py`: `from_cmdstanpy`, `from_nutpie`, `from_blackjax`. The scripts' API is InferenceData **groups**, not the engine name.

| Consumer | Needs |
|---|---|
| Always | `posterior` |
| Divergences / E-BFMI | `sample_stats.diverging` (and `energy` when present) |
| PSIS-LOO | `log_likelihood` from GQ / `vmap` — not `_lupdf` |
| `calibration_check.py` | `posterior_predictive` **and** `observed_data` |
| `psense_summary` | `log_likelihood` and `log_prior` |

BlackJAX exposes `NUTSInfo.is_divergent`. Store it as `diverging` (`from_blackjax` renames `is_divergent` / `divergent`). Omit `sample_stats` and diagnose records **unknown**, not zero divergences.

nutpie does not run generated quantities. A nutpie trace passed straight through has no `log_likelihood` and no PPC until a second GQ / `vmap` pass. LOO and calibration then skip or fail; that is not a clean bill of health.

Do not put a Python `if` or `clip` on a parameter inside a JAX log-density. Constrained decls / `log1p_exp` / a bijector. Hard clips flatten gradients. An observation model whose control flow must iterate until a parameter-dependent stopping rule stays in Stan (Carpenter 2025).
