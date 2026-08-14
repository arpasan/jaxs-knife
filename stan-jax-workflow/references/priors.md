# Prior Selection

## Philosophy

Priors encode domain knowledge. The goal is not to be "non-informative." Every prior needs a justification in `report.md`. If you cannot say why a prior is reasonable, it is not a good prior.

Weakly informative is the default: rule out nonsense, do not strongly favor one reasonable value.

Never `normal(0, 1000)` or `cauchy(0, 100)`. Those are not "flat"; they put mass on absurd scales and wreck HMC.

## Defaults (Stan syntax; JAX uses the same families)

| Parameter | Prior | Notes |
|---|---|---|
| Unbounded location | `normal(0, 2.5)` after standardizing predictors | Raw scale: `sigma ≈ range / 4` |
| Intercept | `normal(mean(y), 2 * sd(y))` | Center on the data scale |
| Scale / SD | `exponential(1)` or `gamma(2, 2)` | Avoid HalfCauchy / HalfFlat on hierarchical scales |
| Proportion | `beta(2, 2)` | Or `logistic` on unconstrained |
| Correlation | `lkj_corr(2)` | `eta=1` uniform; `eta=2` pulls to identity |
| Count rate | `gamma` or `lognormal` | Positive |
| Student-t ν | `gamma(2, 0.1)` | Keeps ν out of the degenerate tail |
| Ordered cutpoints | `ordered` vector + normal | Identifiability |

## Prior predictive (mandatory, before MCMC)

Stan: `generated quantities` with `*_rng` and no conditioning on `y`, or a dummy-data pass. JAX: `vmap` the observation model over prior draws of unconstrained parameters (after `constrain`).

If prior predictive values are impossible (negative blood pressure, billion-dollar daily spend), fix priors first.

## Hierarchical scales

Flat or HalfCauchy scales create the Neal funnel. Prefer `exponential` / `gamma`. Start **non-centered** when groups are small; switch to centered only when groups are data-rich (see [hierarchical.md](hierarchical.md)).

## Sparsity

Many predictors, few expected signals: regularized (Finnish) horseshoe, not a shared `normal`. Horseshoe geometry is hard; raise `adapt_delta` and expect to reparameterize.
