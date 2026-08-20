# Model Comparison

Compare genuinely different assumptions, not for stepwise variable selection. LOO is not a test of “effect ≠ 0” ([CV-FAQ](https://users.aalto.fi/~ave/CV-FAQ.html)).

Prefer **PSIS-LOO** over WAIC. Match the CV partition to the predictive task (LOO vs leave-one-group-out vs leave-future-out). That partition is the **structure of `log_lik`**: leave-one-group-out needs one summed entry per group, not one per row. Keep the heading honest.

```python
comparison = az.compare({"m1": idata_1, "m2": idata_2})
```

Each InferenceData needs pointwise `log_likelihood` from the fully normalized `_lpdf` / `logpdf`, not `_lupdf` or `~` (those drop constants and make cross-model ELPD incomparable).

| ΔELPD vs dSE | Action |
|---|---|
| < 2 × dSE | Indistinguishable; prefer the simpler model |
| 2–4 × dSE | Moderate; use domain knowledge |
| > 4 × dSE | Stronger evidence for the better ELPD |

If Pareto-k is bad, do not compare ELPDs. Fix the model or use K-fold. Moment matching needs a callable log density and unconstrained draws — a CmdStanPy CSV does not carry those. On the Stan path, K-fold is the actionable next step.

**Stacking** when there is no winner (Yao et al.). Report weights, not a fake champion.

WAIC: only if you cannot compute PSIS-LOO. Say so.

A Bayes factor can move by orders of magnitude while the posterior
barely does. Report posterior movement, not a BF, unless the user
asked for a marginal likelihood.
