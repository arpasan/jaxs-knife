# Mini-normal — Bayesian Analysis Report

## Executive Summary

The mean is about 0.25 (50% HDI near the typical value; 94% HDI covers a wider range). There is a high posterior probability the mean is positive. Do not treat any interval width as magic.

## Prior Predictive Check

Prior predictive draws were generated before sampling. They spanned a plausible range around the observed outcomes.

## Sampling and Convergence

Max R-hat was 1.002 (threshold ≤ 1.01). Divergences: 0. The posterior may be interpreted.

## Posterior

| Parameter | Mean | 50% HDI | 94% HDI |
|-----------|------|---------|---------|
| mu | 0.25 | [0.10, 0.38] | [-0.05, 0.55] |
| sigma | 1.05 | [0.95, 1.18] | [0.88, 1.32] |

## Limitations and Threats

1. The likelihood is a single normal; heavier tails were not compared.

## Suggested Next Steps

All diagnostics are within acceptable bounds.
