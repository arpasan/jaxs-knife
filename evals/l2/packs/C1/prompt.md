`y` in data.csv is a stored measurement in device units.

The logger writes a sample only when the value is at least 20.0.
Values below 20.0 were seen by the sensor and dropped; nothing
about them was recorded.

Estimate the mean and the standard deviation of the ambient
process. Name them `mu` and `sigma` in the saved posterior.

Prior predictive checks before sampling. Diagnose R-hat, ESS, and
divergences. Write report.md with 50% and 94% HDIs, probability
language (no p-values), and a limitations section. Save posterior
draws. Put posterior predictive draws in generated quantities or a
vmap, not a numpy rewrite of the likelihood.
