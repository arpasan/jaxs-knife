`dose`, `n`, and `y` (deaths) in data.csv are a binomial dose–response.

Fit a logistic model. The decision quantity is the dose at which the death
probability is 1/2. Compute that functional in generated quantities or a
vmap, not a numpy rewrite of the likelihood after sampling.

Prior predictive checks before sampling. Diagnose R-hat, ESS, and
divergences. Write report.md with 50% and 94% HDIs, probability language
(no p-values), and a limitations section. Save posterior draws.
