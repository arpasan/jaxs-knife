Write a JAX log-density with an explicit constrain step and Jacobian, sample
with BlackJAX NUTS, convert draws to ArviZ, and diagnose. Posterior predictive
via vmap. report.md with 50% and 94% HDIs and limitations. Name the location
`mu` and the scale `sigma` in the saved posterior. Do not transpile Stan to
JAX.
