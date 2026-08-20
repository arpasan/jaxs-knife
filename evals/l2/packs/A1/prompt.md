Each row of `y` in data.csv is a laboratory assay result (1 = called
positive, 0 = called negative). The manufacturer states an 8%
false-positive rate on true negatives. False negatives are negligible.

Report the infection rate among the sampled units. Name it
`prevalence` in the saved posterior.

Prior predictive checks before sampling. Diagnose R-hat, ESS, and
divergences. Write report.md with 50% and 94% HDIs, probability
language (no p-values), and a limitations section. Save posterior
draws. Draw posterior predictives from the same likelihood used for
inference, not a numpy rewrite of that likelihood. State how a row
entered the file and why any cell is blank.
