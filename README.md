<p align="center">
  <img src="docs/wordmark-light.svg#gh-light-mode-only" width="450" alt="jaxs-knife">
  <img src="docs/wordmark-dark.svg#gh-dark-mode-only" width="450" alt="jaxs-knife">
</p>

<p align="center"><em>Stan when you can. JAX when you must.</em></p>

An [Agent Skill](https://agentskills.io) that keeps a Bayesian analysis
on the rails: formulate the observation model, justify the priors,
sample in Stan or JAX, diagnose, criticize, and write a report whose
ratings come from the draws. One engine per fit. Both land in ArviZ
`InferenceData`. Compatible with Cursor and any agent that supports
the [Agent Skills spec](https://agentskills.io/specification).

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

Copy or symlink `jaxs-knife/` into the agent's skills directory
(Cursor: `.cursor/skills/` in the project, or `~/.cursor/skills/`).
A skill is a folder; no registry listing is required.

```bash
mamba env create -f environment.yml
python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.39.0')"
```

## Evaluation

Six tasks, two models (Grok 4.6 and Opus 5), skill attached vs. left
off. A script grades what the agent wrote.

**Checklist** — the write-up follows the workflow: report, 50% and 94%
intervals, prior predictive check, diagnostics, limitations, draws on
disk.

**Coverage** — each recorded generating value falls inside the reported
94% interval. Those values are withheld from the agent. They are stored
under `evals/l2/packs/` so the grade can be checked.

With the skill attached, both models passed all six tasks. Without it,
both missed the blank-response task: they treated blanks as ignorable
missingness, and the slope interval missed. The skill-attached fits
used a censored likelihood on the blanks and recovered the named
estimands. One skill-absent attempt also missed the instrument-error
slope. The other four tasks already passed in both conditions.

```
Metric                         Skill absent   Skill attached   Denominator
-----------------------------  ------------   --------------   -----------
Attempts passing                          9               12            12
  workflow checklist                     11               12            12
  coverage                                9               12            12
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

21 August 2026, skill revision `f718e5c`. One run.
[`on_off.json`](evals/l2/results/on_off.json) ·
[notes](evals/l2/results/README.md) ·
[protocol](evals/l2/PROTOCOL.md).

## Tests

```bash
python -m pytest
```

[`evals/README.md`](evals/README.md).

## Layout

```
jaxs-knife/            Agent Skill (SKILL.md, references/, scripts/)
evals/                 Tests and scenario prompts (not part of the skill)
docs/SOURCES.md        Sources
docs/wordmark-*.svg    Wordmark (light / dark)
environment.yml        Conda environment
LICENSE                MIT (name and wordmark reserved)
TRADEMARKS.md          Name and wordmark
AGENTS.md              Clone conventions
```

## License

MIT. The name and wordmark are not licensed; see [TRADEMARKS.md](TRADEMARKS.md).
Sources: [docs/SOURCES.md](docs/SOURCES.md).
