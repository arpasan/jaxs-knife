`y` in data.csv is a count. The sample variance is larger than the sample
mean. Fit a Bayesian count model. Compare a Poisson likelihood to an
overdispersed alternative (negative binomial or equivalent). Prefer the
simpler model only if the ELPD difference is smaller than twice its standard
error.

Prior predictive checks before sampling. Diagnose R-hat, ESS, and
divergences. Write report.md with 50% and 94% HDIs, probability language
(no p-values), and a limitations section. Save posterior draws. Put posterior
predictive draws in generated quantities or a vmap, not a numpy rewrite of
the likelihood.
