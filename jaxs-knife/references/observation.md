# Observation and recording

A likelihood is a story about how the file was produced, not only about
how a marked unit generates a number. Before priors, write the inclusion
rule. If you cannot say which units appear in the table and why, the
sampling model is incomplete (Gelman and Vehtari, BDA3, ch. 8;
Kumar, Martin, and Lao, BMCP §9.3.1).

Ignorability is a claim, not a default. It fails when the chance that a
row is written depends on the outcome, on a correlate of the outcome, or
on a selection process that the columns do not record.

## Ask, then encode

1. What mechanism put a row in the file?
2. What was seen and discarded, and is the discarded count known?
3. Is a covariate the true regressor, or an instrument reading?
4. Is a label the true class, or an assay with a stated error rate?
5. Why is a cell blank? Dropping incomplete rows is an inclusion rule.

The observation model is part of the generative story. It is not a
footnote after a complete-data GLM.

## Truncation, censoring, selection, blanks

These are different likelihoods. Do not interchange the names. Do not
treat a blank as a license to impute from the untruncated density.

| Mechanism | What is in the file | Likelihood |
|---|---|---|
| Truncation | Only units that passed a rule; discarded count unknown | Density of the retained variable, divided by the probability of retention |
| Censoring | A mark that the value fell outside a limit; *n* known (the row is still there) | Point density for uncensored rows; tail probability (`*_lcdf` / `*_lccdf`) for censored or blank-at-a-limit rows |
| Selection / length bias | Units intercepted with probability that depends on size or duration | Reweight by the inclusion probability; a complete-data density on the retained slice is the wrong model |
| Item missingness | Some cells blank; the row may still be present | `dropna` is an inclusion rule. Write *why* the cell is blank before choosing a repair |

Stan has `*_lcdf` / `*_lccdf` and truncation syntax `T[L,U]`. JAX has
neither. In JAX, write the tail (`log_ndtr`, a quadrature, or an
equivalent) inside the log-density. Do not drop the normalizing term
because the array only contains retained rows.

A pile-up at a cutoff is a clue, not a prerequisite. Length-biased
sampling can leave no visible floor and still be non-ignorable.

## Blank cells are not one model

A blank is a recording event. The repair depends on *why* it is blank.
These three programs are not substitutes.

| If the blank is | Then | Not this |
|---|---|---|
| Independent of the missing value given recorded columns (MAR) | A complete-case likelihood for `y \| x` is valid. Imputing the blanks from that same `y \| x` returns the same posterior for the regression as deleting the rows | A tail probability. Imputation does not add information about the line |
| A reporting limit, saturation, or other ceiling/floor on the *response* | Censoring. The blank (or piled-up) rows contribute `*_lccdf` / `*_lcdf` at the limit. Then **refit**. The process intercept and slope are not the complete-case line | A latent `y_mis ~ normal(alpha + beta * x, sigma)` draw. That is MAR imputation. A logistic `P(blank \| x)` or `P(blank \| x, y)` grid is a selection model, not a censored likelihood |
| Missing because of what the value would have been, with no limit in the file | MNAR. The missing entry is a parameter *and* the observation indicator has a likelihood that depends on that value | Deleting the rows and calling it MAR |

Naming the next likelihood in a limitations paragraph is not a fit. If
criticism says the right program is a tail probability, that program is
the next commit.

## How to tell a response ceiling from MAR on `x`

Plot the recorded values. Compare their support to the posterior
predictive of a complete-data model.

- Recorded `y` stops at a hard max (or min) while `y_rep` routinely
  walks past that edge: that is a reporting-limit fingerprint. The next
  model is censoring (`*_lccdf` / `*_lcdf`), not another MAR imputation
  and not a logistic missingness grid.
- Blanks track a fully observed covariate, and recorded `y` still spans
  the residual range at those `x`: MAR given that covariate is a live
  option. Complete-case and MAR imputation then agree, as they should.
- `x` and `y` can be collinear, so a missingness pattern that rises in
  `x` can look like a missingness pattern that rises in `y`. The
  *support* of recorded `y` vs. `y_rep` is the fork, not the correlation
  of the blank indicator with `x`.

Do not stop after writing “if this were a limit, the slope would
change.” Fit the censored program or say you refused to.

## Predictors with reported error

If a covariate arrives with a reported standard error, the printed
value is not the cause. The regression is on the unmarked true value;
the instrument error is a measurement likelihood. A naive `y ~ x_obs`
is attenuated and over-confident. Posterior predictive checks and
LOO on that naive model can look clean: they score a different
question.

Do not invent an error model when the file has no precision, replicate,
or instrument statement. Do not ignore one that is sitting in the
columns.

## Imperfect labels

A stated false-positive or false-negative rate is part of the
observation model. The sampling probability of a positive call is not
the prevalence. Treating the calls as truth biases the rate toward the
assay's false-positive floor.

## Recovery does not bless the observation model

Fake-data recovery simulates from *this* program. If the program
ignores truncation, measurement error, or assay error, recovery will
happily return the parameters of the wrong story. Use recovery to
check that the pipeline can see a named estimand. Use the inclusion
rule, not a successful recovery, to decide whether the observation
model is the one that generated the file.

A recovery that blanks rows from the *assumed* MAR or logistic rule
cannot validate that rule. It only shows the fitted program is
internally consistent.

PPC and PSIS-LOO are likewise silent on a shared observation misspecification:
they compare the fitted model to the recorded slice, not to the process
that produced the slice. An exception is a support mismatch (recorded
extrema vs. replicated extrema): that is a clue about the inclusion
rule, and it obliges a refit, not a footnote.

If the file is a sample and the estimand is a population share, say so.
The sample is not the population; poststratify or restrict the claim.
