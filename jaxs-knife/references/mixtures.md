# Mixtures and discrete latents

Stan cannot declare discrete parameters. Marginalize. Do not sample a latent class, and do not plug `E[z]` into a nonlinear likelihood.

## Finite mixtures

Weights: `simplex[K] theta` with a Dirichlet prior.
Likelihood: per-observation `log_mix` (or `log_sum_exp` of component `*_lpdf` terms; use `_lpdf`, not `_lupdf`).
When component summaries matter, `ordered[K]` (or `positive_ordered`) in the program is the default: bad component R-hat with `lp__` R-hat ≈ 1 is label switching — use `ordered`.
Sorting after sampling does not constrain the sampler.
Post-hoc relabel only for 1-D locations.
If only the posterior predictive matters, swapped labels are harmless.

## JAX

Put the order constraint **inside** `constrain` (ordered transform + Jacobian) when component summaries matter. Do not `jnp.sort` the posterior after the fact; post-hoc relabel only for 1-D locations.

## Censoring, truncation, and misclassification

A two-component mixture is not the same object as a censored or
truncated sample, and it is not an imperfect assay.

- Censoring and truncation are observation rules. Encode them with
  tail probabilities or a retention factor, not with a spare mixture
  component "for the missing part." See [observation.md](observation.md).
- A stated false-positive or false-negative rate is a misclassification
  likelihood on the label, not a mixture on the latent class you wished
  you had sampled.
- JAX has no `*_lccdf`. Write the tail in the log-density. Do not
  approximate truncation by dropping rows and fitting a complete-data
  mixture.

## Discrete membership

Integrate out. A relaxed one-hot is not a substitute.
