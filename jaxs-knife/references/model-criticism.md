# Model Criticism

Convergence says the sampler worked. Criticism asks whether the model is any good.

## Posterior predictive

Simulate data from the fitted model and compare to `y`.

- Stan: `generated quantities` + `*_rng`.
- JAX: `vmap` the observation model over posterior draws.

Do **not** reimplement the likelihood in numpy after a Stan fit.

Look at shape, spread, tails, and decision-relevant functionals — not a posterior-predictive *p*-value as a hypothesis test (Vehtari BDA3 notes).

Decision functionals (contrasts, ratios, inverse-link doses) belong in `generated quantities` or `vmap`, computed from the same parameters as the fit. Do not reconstruct them in numpy after the fact.

When the PPC misses the data, pick the repair from the **observed** misfit, then change one thing and refit.

| Observed misfit | First repair |
|---|---|
| Tails too light | Student-t / heavier residual |
| Counts overdispersed | If rows have unequal exposure (person-years, area, trials), put `log(exposure)` in the mean first. Only then `neg_binomial_2` |
| Groups ignored | Hierarchy (non-centered if weakly identified) |
| Mean systematically off | Likelihood family or link |
| Mixture / labels unstable | [mixtures.md](mixtures.md) |
| Slope attenuated; PPC still fine | Predictor measured with error ([observation.md](observation.md)) |
| Mean off a known retention rule | Truncation / selection, not a new residual family |

Do not stack three likelihood changes in one refit.

## Predict at new X

New covariates go through `generate_quantities` (Stan) or `vmap` (JAX) on the fitted draws. Do not rebuild a likelihood in numpy. There is no NLPD leaderboard and no `test.csv` ritual — if the user did not hold out a prediction set, do not invent one.

## PSIS-LOO (single-model)

```python
loo = az.loo(idata, pointwise=True)
```

Requires pointwise `log_likelihood` on the InferenceData (Stan `log_lik` in GQ, or JAX `vmap`).

| Pareto k | Action |
|---|---|
| < 0.5 | Trust LOO |
| 0.5–0.7 | Investigate those points |
| ≥ 0.7 | K-fold; do not trust that ELPD. Moment matching needs a callable density, not a CmdStan CSV |

`p_loo` ≫ parameter count → misspecification or priors too weak.

LOO-PIT (`az.plot_loo_pit`) is the calibration view of the predictive
distribution. A systematic S-shape or slope is misfit in spread or
location. It is not the same object as
`scripts/calibration_check.py`, which scores nominal HDI coverage of
this dataset's observations.

## Calibration

`scripts/calibration_check.py` compares empirical HDI coverage of *this* dataset's observations under the posterior predictive to the nominal 94% (or the interval you actually reported). That is a PPC diagnostic on one fit. It is not a LOO-PIT, and it is not a calibration or coverage claim. Those require repeated datasets (SBC or simulated replications).

| Mean coverage Δ | Diagnosis |
|---|---|
| \|Δ\| ≤ 0.02 | Well-calibrated |
| Δ > 0.02 | Under-confident (too uncertain) |
| Δ < −0.02 | Over-confident (too certain) |

## Fake-data recovery

Before the real fit: simulate from the same program at known parameter values, refit, and confirm the **named estimand** falls in the **interval you will report** (50% typical + 94% range). The recovery fit must pass the same diagnose gates as a real fit. One recovery is a check that the pipeline can see the truth; it is not a coverage claim. Coverage needs repeated datasets (SBC or simulated replications).

Recovery cannot bless the observation model. If the program omits
truncation, measurement error, or an assay error rate, a successful
recovery only shows that the wrong story is internally consistent. The
inclusion rule is decided before this step ([observation.md](observation.md)).

## SBC

For a **new** implementation (new JAX density, new constraint). Too expensive for every routine GLM. Histogram of posterior ranks of known parameters should be uniform.
