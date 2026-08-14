# Model Criticism

Convergence says the sampler worked. Criticism asks whether the model is any good.

## Posterior predictive

Simulate data from the fitted model and compare to `y`.

- Stan: `generated quantities` + `*_rng`.
- JAX: `vmap` the observation model over posterior draws.

Do **not** reimplement the likelihood in numpy after a Stan fit.

Look at shape, spread, tails, and decision-relevant functionals — not a posterior-predictive *p*-value as a hypothesis test (Vehtari BDA3 notes).

## PSIS-LOO (single-model)

```python
loo = az.loo(idata, pointwise=True)
```

Requires pointwise `log_likelihood` on the InferenceData (Stan `log_lik` in GQ, or JAX `vmap`).

| Pareto k | Action |
|---|---|
| < 0.5 | Trust LOO |
| 0.5–0.7 | Investigate those points |
| ≥ 0.7 | K-fold or moment matching; do not trust that ELPD |

`p_loo` ≫ parameter count → misspecification or priors too weak.

## Calibration

`scripts/calibration_check.py` compares empirical HDI coverage to the nominal 94% (or the interval you actually reported).

| Mean coverage Δ | Diagnosis |
|---|---|
| \|Δ\| ≤ 0.02 | Well-calibrated |
| Δ > 0.02 | Under-confident (too uncertain) |
| Δ < −0.02 | Over-confident (too certain) |

## SBC

For a **new** implementation (new JAX density, new constraint). Too expensive for every routine GLM. Histogram of posterior ranks of known parameters should be uniform.
