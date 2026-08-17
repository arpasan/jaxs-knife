`y` in data.csv is a two-component normal mixture.

Do not sample a discrete component indicator; marginalize (`log_mix` or
log-sum-exp). Use an ordered constraint so the two means are identified.
Name the ordered means `mu1` and `mu2` (or a length-2 `mu`) and the
first-component weight `weight` in the saved posterior.
Prior predictive checks before sampling. Diagnose R-hat, ESS, and
divergences. Write report.md with 50% and 94% HDIs, probability language
(no p-values), and a limitations section. Save posterior draws. Put
posterior predictive draws in generated quantities or a vmap, not a numpy
rewrite of the likelihood.
