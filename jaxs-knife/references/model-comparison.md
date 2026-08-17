# Model Comparison

Compare genuinely different assumptions, not for stepwise variable selection. LOO is not a test of “effect ≠ 0” ([CV-FAQ](https://users.aalto.fi/~ave/CV-FAQ.html)).

Prefer **PSIS-LOO** over WAIC. Match the CV partition to the predictive task (LOO vs leave-one-group-out vs leave-future-out).

```python
comparison = az.compare({"m1": idata_1, "m2": idata_2})
```

Each InferenceData needs pointwise `log_likelihood`.

| ΔELPD vs dSE | Action |
|---|---|
| < 2 × dSE | Indistinguishable; prefer the simpler model |
| 2–4 × dSE | Moderate; use domain knowledge |
| > 4 × dSE | Stronger evidence for the better ELPD |

If Pareto-k is bad, do not compare ELPDs. Fix the model or use K-fold.

**Stacking** when there is no winner (Yao et al.). Report weights, not a fake champion.

WAIC: only if you cannot compute PSIS-LOO. Say so.
