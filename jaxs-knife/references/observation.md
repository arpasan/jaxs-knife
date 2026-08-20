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

The observation model is part of the generative story. It is not a
footnote after a complete-data GLM.

## Truncation, censoring, selection

These are different likelihoods. Do not interchange the names.

| Mechanism | What is in the file | Likelihood |
|---|---|---|
| Truncation | Only units that passed a rule; discarded count unknown | Density of the retained variable, divided by the probability of retention |
| Censoring | A mark that the value fell outside a limit; *n* known | Point density for uncensored rows; tail probability (`*_lcdf` / `*_lccdf`) for censored rows |
| Selection / length bias | Units intercepted with probability that depends on size or duration | Reweight by the inclusion probability; a complete-data density on the retained slice is the wrong model |

Stan has `*_lcdf` / `*_lccdf` and truncation syntax `T[L,U]`. JAX has
neither. In JAX, write the tail (`log_ndtr`, a quadrature, or an
equivalent) inside the log-density. Do not drop the normalizing term
because the array only contains retained rows.

A pile-up at a cutoff is a clue, not a prerequisite. Length-biased
sampling can leave no visible floor and still be non-ignorable.

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

PPC and PSIS-LOO are likewise silent on a shared observation misspecification:
they compare the fitted model to the recorded slice, not to the process
that produced the slice.
