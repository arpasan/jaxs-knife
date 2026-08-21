<p align="center">
  <img src="docs/wordmark-light.svg#gh-light-mode-only" width="450" alt="jaxs-knife">
  <img src="docs/wordmark-dark.svg#gh-dark-mode-only" width="450" alt="jaxs-knife">
</p>

<p align="center"><em>Stan when you can. JAX when you must.</em></p>

An [Agent Skill](https://agentskills.io) for Bayesian modeling
with Stan (CmdStanPy / nutpie) or JAX (log-density + BlackJAX). One engine
per fit. Both land in ArviZ `InferenceData` and share one diagnose →
calibrate → report pipeline. The steps are informed by the Gelman–Vehtari Bayesian workflow;
see [docs/SOURCES.md](docs/SOURCES.md).

Compatible with Cursor and any agent that supports the
[Agent Skills spec](https://agentskills.io/specification).

## What it does

1. Formulate the generative story, including how a row got into the file
2. Specify weakly informative, justified priors
3. Implement in Stan or JAX (see the engine table in the skill)
4. Prior predictive checks before sampling
5. Fake-data recovery, then inference (nutpie on Stan when the toolchain
   works; otherwise CmdStanPy; BlackJAX on a JAX log-density)
6. Convergence diagnostics (R-hat, ESS, divergences, energy, treedepth)
7. Model criticism (PPC in generated quantities or `vmap`, PSIS-LOO, calibration)
8. Prior sensitivity when conclusions are decision-relevant
9. Model comparison (PSIS-LOO, stacking)
10. A canonical `<slug>/report.md` whose ratings are computed from
    diagnostics, not asserted

## Install

A skill is a folder. Copy or symlink `jaxs-knife/` into the agent's
skills directory. No registry listing is required for the skill to
load. For Cursor: `.cursor/skills/` in the project, or
`~/.cursor/skills/`. Do this only when you mean the skill to attach
automatically.

```bash
mamba env create -f environment.yml
python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.39.0')"
```

## Evaluation

Six isolated tasks under `evals/l2/packs/` were run once with the skill
absent from the host catalog and once with the skill attached, by two
models (Grok 4.6 and Opus 5): 24 attempts. The two attempts on a task
are different models, not two copies of one model. A deterministic
script grades the files the agent left behind.

Observed counts, 21 August 2026, skill revision
`f718e5c882e7dffb645ea7d68daac1ed637417e8`:

```
Metric                         Skill absent   Skill attached   Denominator
-----------------------------  ------------   --------------   -----------
Attempts passing                          9               12            12
  component: workflow checklist          11               12            12
  component: recorded-value coverage      9               12            12
Tasks passed by both models               4                6             6

Task                                      Skill absent   Skill attached
----------------------------------------  ------------   --------------
Predictor measured with error                    1/2              2/2
Grouped observations and new group               2/2              2/2
Assay with stated false-positive rate            2/2              2/2
Two-component sample                             2/2              2/2
Positive sample with stated quartiles            2/2              2/2
Predictor with blank responses                   0/2              2/2
```

An attempt passes when the write-up meets a fixed workflow checklist
(report, 50% and 94% intervals, prior predictive check, diagnostics,
limitations, draws on disk) and each recorded generating value lies in
the reported 94% interval. Those values are withheld from the agent's
directory; they remain in this repository so the grade is checkable.
The checklist scores the workflow steps, not the library used.

This is one run. Three of twelve paired attempts differed, all in the
same direction. The skill-absent fits used PyMC; the skill-attached
fits used Stan via CmdStanPy — the two are not separable here.
Four of the six tasks already passed in both conditions. The
blank-response observation guidance was written in this same
revision. Machine-readable scores:
[`evals/l2/results/on_off.json`](evals/l2/results/on_off.json).
Notes and limits:
[`evals/l2/results/README.md`](evals/l2/results/README.md).
Design and grading predicates:
[`evals/l2/PROTOCOL.md`](evals/l2/PROTOCOL.md).

## Tests

```bash
python -m pytest
```

Fast checks live under `evals/l0` and `evals/l2`. Live NUTS smoke tests
live under `evals/smoke`. See [`evals/README.md`](evals/README.md).

## Layout

```
jaxs-knife/            Agent Skill (SKILL.md, references/, scripts/)
evals/                 Tests and scenario prompts (not part of the skill)
docs/SOURCES.md        Literature the skill encodes
docs/wordmark-*.svg    Wordmark (light / dark)
environment.yml        Conda environment
LICENSE                MIT (name and wordmark reserved)
TRADEMARKS.md          Name and wordmark
AGENTS.md              Repo conventions
```

## License

MIT. The name and wordmark are not licensed; see [TRADEMARKS.md](TRADEMARKS.md).
