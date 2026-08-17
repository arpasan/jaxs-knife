# Reporting

After every full run, write `<slug>/report.md`. Static sentences stay verbatim. Fill `<placeholders>`. Ratings come from `scripts/check_diagnostics.py`.

## Results folder

```
<slug>/
├── inference_data.nc
├── trace.png
├── forest.png
├── posterior_predictive.png
├── prior_predictive.png
├── summary.csv
├── diagnostics.json
├── calibration.json
├── check_report.json
└── report.md
```

Optional: `pit_ecdf.png`, `psense.json`, CmdStan `diagnose` text.

## Interval policy

No width is magic. Default: **50% typical + 94% range** (HDI). 80% HDI is notebook ink, not the scientific default. Do not mix HDI with ArviZ bare ETI defaults without saying so.

A single 94% HDI is a statement about this posterior, not a calibration claim. Coverage and calibration require repeated datasets (SBC or simulated replications).

Report Monte Carlo standard error next to every printed posterior number. Gate ESS on the interval you print.

## Language

Probability language. Never “significant,” “rejected,” or p-values. Posterior **mean** of predictive probabilities, never median.

## Template

````markdown
# <Analysis Title> — Bayesian Analysis Report

## Executive Summary

<2–3 sentences: finding, 50% and 94% HDI, most important caveat. Lead with the substance.>

## Data and Question

| | |
|---|---|
| Source | <source> |
| Sample size | <N> |
| Key variables | <list> |
| Question | <one sentence> |
| Engine | <Stan CmdStanPy / Stan nutpie / JAX BlackJAX / both> |

## Model Specification

**Generative story.** <1–3 sentences.>

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| <param> | <prior> | <why> |

## Prior Predictive Check

**Assessment:** <do prior draws span a plausible range?>

## Sampling and Convergence

| Diagnostic | Value | Threshold | Status |
|------------|-------|-----------|--------|
| Max R-hat | <> | ≤ 1.01 | <> |
| Min ESS (bulk) | <> | ≥ 100 × n_chains | <> |
| Min ESS (tail) | <> | ≥ 100 × n_chains | <> |
| Divergences | <> | 0 | <> |

**Assessment:** <paste `check_diagnostics` convergence sentence. If divergences > 0, state that the posterior is not interpreted.>

## Posterior

| Parameter | Mean | SD | MCSE | 50% HDI | 94% HDI | P(>0) |
|-----------|------|----|------|---------|---------|-------|
| <param> | <> | <> | <> | <> | <> | <> |

**Substantive interpretation.** <domain units; no frequentist language.>

## Posterior Predictive Check

**Assessment:** <does PPC cover the data? systematic miss?>

## Calibration

**Assessment:** <paste calibration sentence.>

## [IF MODEL_COMPARISON] Model Comparison

| Model | ELPD | SE | ΔELPD | Weight |
|-------|------|----|-------|--------|
| <a> | <> | <> | <> | <> |

**Assessment:** <ΔELPD vs 2 × dSE; prefer simpler if indistinguishable.>

## [IF SENSITIVITY] Prior Sensitivity

**Assessment:** <which parameters flag; does the conclusion depend on the prior?>

## Limitations and Threats

Mandatory. Rank by severity. Assumption, bias direction, what would resolve it.

1. <threat>

## Suggested Next Steps

<`check_diagnostics.py` `next_steps`, plus problem-specific context.>

## Appendix

<ArviZ summary. Seed. CmdStan / CmdStanPy / nutpie / JAX / BlackJAX / ArviZ versions. JAX float precision (`jax_enable_x64` on or off). Pathfinder labeled if used.>
````
