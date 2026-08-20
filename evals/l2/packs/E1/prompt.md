`y` in data.csv is a response. `x` is a reading from a laboratory
instrument; `x_se` is that instrument's reported standard error for
the same reading.

Estimate the intercept and the slope of the process that generates
`y` from the unmarked true value of the quantity the instrument is
measuring. Name them `alpha` and `beta` in the saved posterior.

Prior predictive checks before sampling. Diagnose R-hat, ESS, and
divergences. Write report.md with 50% and 94% HDIs, probability
language (no p-values), and a limitations section. Save posterior
draws. Draw posterior predictives from the same likelihood used for
inference, not a numpy rewrite of that likelihood.
