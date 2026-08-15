# standoff-bayes

An opinionated [Agent Skill](https://agentskills.io) for Bayesian modeling
with Stan (CmdStanPy / nutpie) and JAX (log-density + BlackJAX). Both engines
land in ArviZ `InferenceData` and share one diagnose → calibrate → report
pipeline.

Compatible with Cursor and any agent that supports the
[Agent Skills spec](https://agentskills.io/specification).

## What it does

1. Formulate the generative story
2. Specify weakly informative, justified priors
3. Implement in Stan or JAX (see the engine table in the skill)
4. Prior predictive checks before sampling
5. Inference (nutpie on Stan when the toolchain works; otherwise CmdStanPy;
   BlackJAX on a JAX log-density)
6. Convergence diagnostics (R-hat, ESS, divergences, energy, treedepth)
7. Model criticism (PPC in generated quantities or `vmap`, PSIS-LOO, calibration)
8. Prior sensitivity when conclusions are decision-relevant
9. Model comparison (PSIS-LOO, stacking)
10. A canonical `<slug>/report.md` whose ratings come from scripts, not vibes

## Install

```bash
mamba env create -f environment.yml
python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.39.0')"
```

Copy or symlink `stan-jax-workflow/` into the agent's skills directory
(for Cursor: `.cursor/skills/` in the project, or `~/.cursor/skills/`).

## Tests

```bash
python -m pytest
```

`evals/l0` checks the diagnostic JSON contract on known-good and known-bad
traces. `evals/l2` grades sealed workflow fixtures (no live agent).
`evals/smoke` runs live CmdStanPy and BlackJAX NUTS on the same mini-normal
data.

## Layout

```
stan-jax-workflow/     Agent Skill (SKILL.md, references/, scripts/)
evals/                 Tests and scenario prompts (not part of the skill)
environment.yml        Conda environment
```

## License

MIT
