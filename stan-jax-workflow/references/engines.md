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
3. Prior + likelihood on constrained values.
4. `vmap` the observation model for GQ / PPC.

BlackJAX is a sampler library, not a PPL. You owe it a `logdensity_fn`.

NumPyro is optional when plates help. Override the classic 8-schools HalfCauchy example.

## What not to do

- Transpile Stan → XLA.
- Treat BridgeStan + JAX callbacks as a `jit`/`gpu` Stan path (grads stay C++).
- Report Pathfinder / Laplace as the posterior unless the user asked for an approximation.
- Rewrite a Stan-shaped GLM in NumPyro without a reason the Stan language cannot express.
- Likelihood-free / simulation-based inference (no tractable likelihood).

## Conversion

`scripts/to_inference_data.py`: `from_cmdstanpy`, `from_nutpie`, `from_blackjax`. Downstream scripts never see the engine.
