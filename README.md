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

Each task was assigned two conditions: skill absent from the agent's
workspace, then skill attached. Nothing else changed. One model was
held fixed across both conditions. Each task's prompt and data file
were used as stored under `evals/l2/packs/`. Three independent
attempts were made per task and condition, each in its own directory
with no shared memory. A deterministic script grades the files the
agent left behind. Scores:
[`evals/l2/results/README.md`](evals/l2/results/README.md)
([`on_off.json`](evals/l2/results/on_off.json)).

An attempt passes when the write-up meets a fixed workflow checklist
(report, 50% and 94% intervals, prior predictive check, diagnostics,
limitations, draws on disk) and, when the task records the parameter
values used to generate its data, each value lies in the reported 94%
interval. Those values are withheld from the agent's directory; they
remain in this repository so the grade is checkable.

Two suites share that grader. The first is eight reporting exercises
(regression through a JAX log-density). The second is four science
problems — a mixture, a hierarchical model, a JAX location-scale fit,
and a recording process that drops low values — scored with the same
write-up checklist.

```text
+------------------------+--------------------------+-----------+----------+
| Suite                  | Measure                  | skill off | skill on |
+------------------------+--------------------------+-----------+----------+
| Eight reporting tasks  | attempts passing         |    5 / 24 |  21 / 24 |
|                        | tasks passing 3 of 3     |     0 / 8 |    6 / 8 |
|                        | generating-value covered |   15 / 15 |  15 / 15 |
+------------------------+--------------------------+-----------+----------+
| Four science tasks     | attempts passing         |    6 / 12 |  12 / 12 |
|                        | tasks passing 3 of 3     |     1 / 4 |    4 / 4 |
|                        | generating-value covered |   12 / 12 |  12 / 12 |
+------------------------+--------------------------+-----------+----------+
```

Coverage is scored only on tasks that record generating values (five of
the eight reporting tasks, and all four science tasks). Each such task
was accepted only if a reference interval under the task's own priors
contained the value, so that row is a floor check: it was already
complete without the skill, and attaching the skill did not disturb it.

The difference between conditions is therefore confined to the write-up
and its diagnostics. In this run the model specified and fit these
problems under both conditions; what it omitted when unprompted is the
prior predictive check, the convergence statement, the criticism step,
and the interval discipline that let a reader audit the result. That is
what the skill supplies. It is not a claim of a better posterior. Two
of the eight reporting tasks still have failing attempts with the skill
attached; see the per-task table in the results note.

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
