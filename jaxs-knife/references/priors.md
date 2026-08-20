# Prior Selection

## Philosophy

Priors encode domain knowledge. The goal is not to be "non-informative." Every prior needs a justification in `report.md`. If you cannot say why a prior is reasonable, it is not a good prior.

Weakly informative is the default: rule out nonsense, do not strongly favor one reasonable value.

Never `normal(0, 1000)` or `cauchy(0, 100)`. Those are not "flat"; they put mass on absurd scales and wreck HMC.

## Defaults (Stan syntax; JAX uses the same families)

| Parameter | Prior | Notes |
|---|---|---|
| Unbounded location | `normal(0, 2.5)` after standardizing predictors | Raw scale: `sigma ≈ range / 4` |
| Intercept | `normal(mean(y), 2 * sd(y))` | Complete-data scale only. Do not use on a selected or truncated slice. |
| Scale / SD | `exponential(1)` or `gamma(2, 2)` | Avoid HalfCauchy / HalfFlat on hierarchical scales |
| Proportion | `beta(2, 2)` | Or `logistic` on unconstrained |
| Correlation | `lkj_corr(2)` | `eta=1` uniform; `eta=2` pulls to identity |
| Count rate | `gamma` or `lognormal` | Positive |
| Student-t ν | `gamma(2, 0.1)` | Keeps ν out of the degenerate tail |
| Ordered cutpoints | `ordered` vector + normal | Identifiability |

## Prior predictive (mandatory, before MCMC)

Prefer a `prior_only` data flag in the **same** program so the prior predictive cannot drift from the fitted model. Stan: `generated quantities` with `*_rng` and no conditioning on `y` when `prior_only` is set. JAX: `vmap` the observation model over prior draws of unconstrained parameters (after `constrain`).

If prior predictive values are impossible (negative blood pressure, billion-dollar daily spend), fix priors first.

## Selected or truncated samples

`normal(mean(y), 2 * sd(y))` on the intercept is a complete-data default.
It is the wrong center when rows were kept because of `y` or a correlate
of `y`. The retained-sample mean is a biased slice; putting a tight
prior there can make the interval prior-determined and hide the
selection. Prefer a weakly informative location that does not treat the
observed mean as the process mean (`normal(0, 2.5)` after a stated
scale, or a domain center that does not come from the kept rows).

The same warning applies to empirical scales taken from the kept slice
and then reused as if they described the ambient process. See
[observation.md](observation.md).

## Hierarchical scales

Flat or HalfCauchy scales create the Neal funnel. Prefer `exponential` / `gamma`. Start **non-centered** when groups are small; switch to centered only when groups are data-rich (see [hierarchical.md](hierarchical.md)).

## Sparsity

Many predictors, few expected signals: regularized (Finnish) horseshoe, not a shared `normal`. Horseshoe geometry is hard; raise `adapt_delta` and expect to reparameterize.
