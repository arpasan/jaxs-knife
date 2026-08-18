# Mixtures and discrete latents

Stan cannot declare discrete parameters. Marginalize. Do not sample a latent class, and do not plug `E[z]` into a nonlinear likelihood.

## Finite mixtures

Weights: `simplex[K] theta` with a Dirichlet prior. Component locations: `ordered[K] mu` (or `positive_ordered`) **in the program**. Sorting means after sampling does not constrain the sampler.

Likelihood: `log_mix` or `log_sum_exp` of component `*_lpdf` terms. Use `_lpdf`, not `_lupdf`.

Label switching looks like bad R-hat. More draws will not fix it. If only the posterior predictive matters, swapped labels are harmless. Component-wise summaries are not.

## JAX

Put the order constraint **inside** `constrain` (ordered transform + Jacobian). Do not `jnp.sort` the posterior after the fact.

## Discrete membership

Integrate out. A relaxed one-hot is not a substitute.
