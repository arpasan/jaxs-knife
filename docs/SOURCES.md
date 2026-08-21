# Sources

These works are the sources for the methods encoded in the skill.
None of their authors are affiliated with this project, and none have
reviewed or endorsed it.

The short in-context list stays in `jaxs-knife/SKILL.md`. This page
is the bibliography.

## Workflow and reporting

- Gelman, Vehtari, Simpson, Margossian, Carpenter, Yao, Kennedy,
  Gabry, Bürkner, and Modrák (2020). *Bayesian Workflow*.
  [arXiv:2011.01808](https://arxiv.org/abs/2011.01808).
- Gelman, Carlin, Stern, Dunson, Vehtari, and Rubin (2013).
  *Bayesian Data Analysis*, 3rd ed., especially ch. 8 (missing data
  and observation).
- Vehtari. *Bayesian Data Analysis* course notes (Aalto). Cited in
  the skill as course notes, not as the book.
- Gabry, Simpson, Vehtari, Betancourt, and Gelman (2019).
  Visualization in Bayesian workflow. *Journal of the Royal
  Statistical Society: Series A*, 182(2).
- Martin, Kumar, and Lao (2021). *Bayesian Modeling and Computation
  in Python*, §9.3.1.

## Diagnostics and model criticism

- Vehtari, Gelman, Simpson, Carpenter, and Bürkner (2021).
  Rank-normalization, folding, and localization: an improved \(\hat{R}\)
  for assessing convergence of MCMC.
  *Bayesian Analysis*, 16(2).
- Vehtari, Gelman, and Gabry (2017). Practical Bayesian model
  evaluation using leave-one-out cross-validation and WAIC.
  *Statistics and Computing*, 27(5).
- Vehtari, Simpson, Gelman, Yao, and Gabry (2024). Pareto smoothed
  importance sampling. *Journal of Machine Learning Research*, 25(72).
- Yao, Vehtari, Simpson, and Gelman (2018). Using stacking to average
  Bayesian predictive distributions. *Bayesian Analysis*, 13(3).
- Kallioinen, Paananen, Bürkner, and Vehtari (2024). Detecting and
  diagnosing prior and likelihood sensitivity with power-scaling.
  *Statistics and Computing*, 34(57).
- Betancourt (2017). A conceptual introduction to Hamiltonian Monte
  Carlo. [arXiv:1701.02434](https://arxiv.org/abs/1701.02434).
- Hoffman and Gelman (2014). The No-U-Turn Sampler.
  *Journal of Machine Learning Research*, 15.

## Observation models and geometry

- Rubin (1976). Inference and missing data. *Biometrika*, 63(3).
- Neal (2003). Slice sampling. *Annals of Statistics*, 31(3)
  (the funnel).
- Betancourt and Girolami (2015). Hamiltonian Monte Carlo for
  hierarchical models.
- Talts, Betancourt, Simpson, Vehtari, and Gelman (2018).
  Validating Bayesian inference algorithms with simulation-based
  calibration. [arXiv:1804.06788](https://arxiv.org/abs/1804.06788).
  Cited only for the sentence that one recovery is not coverage; this
  skill does not run SBC.

## Software and implementation notes

- Carpenter, Gelman, Hoffman, Lee, Goodrich, Betancourt, Brubaker,
  Guo, Li, and Riddell (2017). Stan: a probabilistic programming
  language. *Journal of Statistical Software*, 76(1).
- Stan Development Team. *Stan User's Guide* and *Reference Manual*,
  version 2.39 (reparameterization; censored data / `*_lccdf`).
- Carpenter (2025). [It's a JAX, JAX, JAX, JAX World](https://statmodeling.stat.columbia.edu/2025/10/03/its-a-jax-jax-jax-jax-world/)
  (blog post: write the density in JAX; do not transpile Stan to XLA).
- Kumar, Carroll, Hartikainen, and Martin (2019). ArviZ.
  *Journal of Open Source Software*, 4(33), 1143.
- Bradbury et al. (2018). JAX.
- Cabezas et al. (2024). BlackJAX.
- Roualdes, Ward, Carpenter, Axen, and Lee (2023). BridgeStan.
  *Journal of Open Source Software*.
- CmdStanPy and nutpie documentation, matching the versions in
  `environment.yml`.
