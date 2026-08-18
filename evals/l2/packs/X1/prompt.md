`y` in data.csv is a location-scale sample.

Write a JAX log-density and sample with BlackJAX NUTS. The scale must
stay positive. Convert draws to ArviZ and diagnose. Name the location
`mu` and the scale `sigma` in the saved posterior.

Prior predictive checks before sampling. Write report.md with 50% and
94% HDIs, probability language (no p-values), and a limitations section.
Put posterior predictive draws in a vmap, not a numpy rewrite of the
likelihood. Do not transpile Stan to JAX.
