<p align="center">
  <img src="docs/wordmark-light.svg#gh-light-mode-only" width="450" alt="jaxs-knife">
  <img src="docs/wordmark-dark.svg#gh-dark-mode-only" width="450" alt="jaxs-knife">
</p>

<p align="center"><em>Stan when you can. JAX when you must.</em></p>

An opinionated [Agent Skill](https://agentskills.io) for Bayesian modeling
with Stan (CmdStanPy / nutpie) or JAX (log-density + BlackJAX). One engine
per fit. Both land in ArviZ `InferenceData` and share one diagnose →
calibrate → report pipeline.

Compatible with Cursor and any agent that supports the
[Agent Skills spec](https://agentskills.io/specification).

## What it does

1. Formulate the generative story
2. Specify weakly informative, justified priors
3. Implement in Stan or JAX (see the engine table in the skill)
4. Prior predictive checks before sampling
5. Fake-data recovery, then inference (nutpie on Stan when the toolchain
   works; otherwise CmdStanPy; BlackJAX on a JAX log-density)
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

Copy or symlink `jaxs-knife/` into the agent's skills directory
(for Cursor: `.cursor/skills/` in the project, or `~/.cursor/skills/`).
Do this only when you mean the skill to attach automatically.

## Tests

```bash
python -m pytest
```

`evals/l0` checks the diagnostic JSON contract on known-good and known-bad
traces. `evals/l2` grades sealed workflow fixtures; the live skill-on vs
skill-off batch (n = 3, pass^1 and pass^3) is described in
[`evals/l2/PROTOCOL.md`](evals/l2/PROTOCOL.md). `evals/smoke` runs live
CmdStanPy and BlackJAX NUTS on the same mini-normal data.

## Layout

```
jaxs-knife/            Agent Skill (SKILL.md, references/, scripts/)
evals/                 Tests and scenario prompts (not part of the skill)
docs/wordmark-*.svg    Wordmark (light / dark)
environment.yml        Conda environment
```

## License

MIT
