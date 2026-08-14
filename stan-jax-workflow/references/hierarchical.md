# Hierarchical Models

Use when observations nest in interchangeable groups (students in schools, patients in hospitals). Time series are **not** hierarchical — timestamps are ordered.

Partial pooling is the default: small groups shrink; large groups keep their estimate.

## Centered vs. non-centered

This is the usual source of divergences (Neal funnel / 8-schools).

**Centered** (data-rich groups):

```stan
vector[J] theta;
theta ~ normal(mu, tau);
```

**Non-centered** (weakly identified groups — start here):

```stan
vector[J] theta_raw;
theta_raw ~ std_normal();
vector[J] theta = mu + tau * theta_raw;
```

Start non-centered. Switch to centered only if non-centered ESS is poor **and** groups have substantial data (~50+ observations each). Within-group centering of covariates (`u^wgc`) when a group-level slope would otherwise collinear with the intercept.

## JAX equivalent

Sample `theta_raw ~ N(0, 1)` unconstrained; set `theta = mu + tau * theta_raw` in `constrain`. Do not put a HalfCauchy on `tau`.

## WGC and QR

Collinear `X`: QR reparameterization from the Stan User’s Guide. Hierarchical slopes: center predictors **within group** so the intercept remains interpretable.
