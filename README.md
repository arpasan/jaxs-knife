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

```bash
mamba env create -f environment.yml
python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.39.0')"
```

Copy or symlink `jaxs-knife/` into the agent's skills directory
(for Cursor: `.cursor/skills/` in the project, or `~/.cursor/skills/`).
Do this only when you mean the skill to attach automatically.

## Evaluation

The sealed suite is six tasks under `evals/l2/packs/`. Each task is
run in two conditions (skill absent from the host catalog, then the
skill attached) and by two solvers (one Grok 4.6 attempt and one
Opus 5 attempt): 24 jobs. The two attempts in a cell are different
models, not two copies of one model. A deterministic script grades
the files the agent left behind.

Scores are pending. A prior sealed run was withdrawn after the
instrument-error coverage screen was found to pin generating latent
moments. The public file is
[`evals/l2/results/on_off.json`](evals/l2/results/on_off.json)
(`status: not_yet_run`). Notes:
[`evals/l2/results/README.md`](evals/l2/results/README.md).

An attempt passes when the write-up meets a fixed workflow checklist
(report, 50% and 94% intervals, prior predictive check, diagnostics,
limitations, draws on disk) and each recorded generating value lies in
the reported 94% interval. Those values are withheld from the agent's
directory; they remain in this repository so the grade is checkable.
The checklist is the scientific workflow, not an engine fashion test.

Each task accepts a CSV only when a naive interval misses the named
estimand and a reference interval under the task's observation model
covers it. The instrument-error reference estimates unmarked-`x`
moments from the instrument columns. Coverage looks up those named
estimands in the posterior or in generated quantities.

Design and grading predicates: [`evals/l2/PROTOCOL.md`](evals/l2/PROTOCOL.md).

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
docs/wordmark-*.svg    Wordmark (light / dark)
environment.yml        Conda environment
LICENSE                MIT (name and wordmark reserved)
TRADEMARKS.md          Name and wordmark
AGENTS.md              Repo conventions
```

## License

MIT. The name and wordmark are not licensed; see [TRADEMARKS.md](TRADEMARKS.md).
