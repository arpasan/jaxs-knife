# Convergence Diagnostics

If any check fails, do **not** interpret results.

```bash
python scripts/diagnose_model.py --idata <slug>/inference_data.nc --output <slug>/diagnostics.json
```

When the fit is CmdStan, also run `fit.diagnose()` / `cmdstan diagnose` for energy and treedepth. Those are first-class, not optional plots.

## Rank-normalized split R-hat (Vehtari et al. 2021)

| R-hat | Action |
|---|---|
| ≤ 1.01 | Proceed |
| 1.01–1.05 | Run longer; investigate |
| > 1.05 | Do not use the posterior |

## ESS

Bulk (means) and tail (intervals) both matter.

| ESS | Action |
|---|---|
| ≥ 100 × n_chains | Sufficient for most summaries |
| Below that | More draws or reparameterize |

## Divergences

Even a few can bias the posterior. Refuse to interpret. Reparameterize (non-centered, Jacobian, QR), **then** raise `adapt_delta` / `target_accept` to 0.95–0.99. Raising adapt first is a last resort.

## Energy (E-BFMI) and treedepth

E-BFMI < 0.3: momentum resampling is sick — usually a funnel or heavy tails. Treedepth saturation: the integrator is hitting the cap; fix geometry before raising `max_treedepth`.

## Visuals

`az.plot_trace` / `az.plot_rank` with explicit `var_names`. Pair plots near ±1 → identifiability, not "interesting correlation."

## Identifiability

If two components always appear together in the likelihood, only their sum is identified. Merge them or restructure the data. Discrete latents: marginalize; do not plug a soft argmax into a nonlinear likelihood.
