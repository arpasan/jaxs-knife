Fit a weakly informative Bayesian linear model for `y` given `x` in data.csv.

Run a prior predictive check before any MCMC. Diagnose convergence (R-hat,
ESS, divergences). Write report.md with 50% and 94% HDIs, probability
language (no p-values), and a limitations section. Save posterior draws.

Choose Stan or a JAX log-density. Put posterior predictive draws in generated
quantities or a vmap, not a numpy rewrite of the likelihood.
