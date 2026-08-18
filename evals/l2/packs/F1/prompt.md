`y` in data.csv is grouped by `group`. Fit a hierarchical model for
the observations. Report 50% and 94% HDIs for the population mean
`mu` and the group-level scale `tau`.

Prior predictive checks before sampling. Diagnose R-hat, ESS, and
divergences. Write report.md with 50% and 94% HDIs, probability language
(no p-values), and a limitations section. Save posterior draws. Put
posterior predictive draws in generated quantities or a vmap, not a numpy
rewrite of the likelihood.

If the sampler diverges, do not interpret the posterior.
