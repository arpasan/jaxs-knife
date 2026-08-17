# Prior Sensitivity

Power-scaling (Kallioinen et al. 2024) asks whether conclusions move when the prior (or likelihood) is mildly strengthened or weakened, without a full refit. It needs `log_likelihood` **and** `log_prior` on the InferenceData.

Stan: store both in `generated quantities` (or compute `log_prior` from the prior densities on the draws). JAX: `vmap` the prior log-density on constrained parameters.

```python
from arviz_stats import psense_summary
summary = psense_summary(idata)
```

CJS > 0.05 flags sensitivity (~0.3 SD shift in the posterior mean). That is a **flag to document**, not a command to loosen the prior.

| Prior CJS | Likelihood CJS | Meaning |
|---|---|---|
| < 0.05 | < 0.05 | Robust |
| > 0.05 | < 0.05 | Prior-driven; justify or widen |
| < 0.05 | > 0.05 | Likelihood-driven; usually fine |
| > 0.05 | > 0.05 | Prior–data conflict; investigate |

If a flagged prior is genuine domain knowledge, keep it and say so in `report.md`. If the substantive conclusion flips when you widen it, report both runs.
