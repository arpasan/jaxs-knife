`y` in data.csv is grouped by `group`. Group identifiers start at 1.

Report the population mean `mu`, the scale of the group means `tau`,
and the mean for group 1 `theta1` in the saved posterior.

Prior predictive checks before sampling. Diagnose R-hat, ESS, and
divergences. Write report.md with 50% and 94% HDIs, probability
language (no p-values), and a limitations section. Save posterior
draws. Draw posterior predictives from the same likelihood used for
inference, not a numpy rewrite of that likelihood.

If the sampler diverges, do not interpret the posterior.
