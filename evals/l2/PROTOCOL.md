# Isolated skill evaluation

Paired evaluation of the skill: same task, same model, skill off then
skill on, three independent attempts, deterministic grade. The harness
lives under this directory. Live solver trees are not committed.

## Design

| Item | Choice |
|---|---|
| Reporting tasks | S1–S8 under `packs/` (regression through a JAX log-density) |
| Science tasks | M1, F1, X1, C1 (mixture, hierarchical, JAX location-scale, recordings that drop low values) |
| Conditions | skill off, then skill on |
| Attempts per cell | *n* = 3 (independent agents, no shared memory) |
| Model | one model, held fixed; recorded only in local per-cell JSON |
| Prompt | pack `prompt.md` verbatim; no coaching |
| Agent folder | `prompt.md` and `data.csv` only; skill copy only when on |
| Generating values | `meta.json` stays in this repository; never copied into the agent folder |
| Grade | workflow checklist on written files; coverage of recorded generating values when present |
| Headline | attempts passing, tasks passing 3 of 3, and coverage, per condition |
| Judge | no LLM string-judge |

pass^k ([τ-bench](https://arxiv.org/abs/2406.12045)) is the probability
that all *k* independent attempts succeed, C(c,k)/C(n,k). pass^1 is the
ordinary pass rate.

Where a task records generating values, its CSV is accepted only if a
reference 94% interval under the task's own priors contains those
values. Coverage is therefore a floor check on the fit. It is not a
claim that the skill produces a better posterior.

## Tasks

| Id | Prompt (one line) | Generating values recorded |
|---|---|---|
| S1 | linear regression | yes |
| S2 | hierarchical school effects | no |
| S3 | overdispersed counts | no |
| S4 | linear regression with prior sensitivity | yes |
| S5 | positive scale (constraint or Jacobian) | no |
| S6 | binomial dose-response | yes |
| S7 | two-component mixture | yes |
| S8 | JAX log-density | yes |
| M1 | two-component mixture | yes |
| F1 | grouped hierarchical | yes |
| X1 | JAX location-scale | yes |
| C1 | recordings that omit values below a threshold | yes |

Prompts do not name this skill.

## Grading

The workflow checklist (implementation: `band_a.py`) is scored on files
the agent wrote. Copied skill trees and `SKILL.md` subtrees are
excluded. S4 adds an optional `prior_sensitivity_refit` predicate from
pack `meta.json`.

Coverage (implementation: `band_b.py`) is scored when the pack records
generating values. The grader reads `inference_data.nc` when several
NetCDF files are present. Missing parameter names are recorded as
failures.

A cell is discarded if `meta.json`, `rubric.json`, or other gold files
appear in the agent folder: that folder is then no longer a blind
attempt.

## Headline metrics

The published headline is **attempts passing**: the workflow checklist,
and coverage when that task records generating values. The two parts
are also stored separately in `results/on_off.json` as
`checklist_successes` and `coverage_successes`.

A robustness cut reports the checklist with and without the literal
`1.01` R-hat predicate.

## Procedure

1. `python evals/l2/run_trial.py --pack S1 --condition without --n 3`
2. Keep `rep-*.receipt.json` (the manifest of files the attempt could
   see).
3. Three new agents, no shared memory, working directory is the `rep-*`
   folder only, model fixed.
4. `python evals/l2/run_trial.py --pack S1 --condition without --n 3 --grade`
   (or `grade.py` on each `rep-*`).
5. Write the per-cell score JSON with `evals/l2/emit_results.py` into a
   gitignored run directory. Do not commit per-cell JSON.
6. Keep live run trees under gitignored directories, including draws.
   `--wipe` is only for shrinking disk after an explicit decision; it
   is not the default after a batch.
7. Rebuild the public aggregate with `evals/l2/summarize_on_off.py`.

Do not write `eval_metadata.json` beside outputs: a file that names the
condition can contaminate the agent folder and void the cell. Do not
run solvers in a session that has seen this protocol or the generating
values.

## Reproducibility

**Fixed:** pack CSVs, prompts, grader, rubric, and the skill revision
used for the run (`eval_commit` in the public file). Local per-cell
JSON records model id, condition, task, commit SHA, date, success
vector, pass^1, and pass^3.

**Not bit-identical:** agent transcripts and MCMC draws. That is why
*n* = 3 and why coverage is “value inside the reported interval,” not
“same posterior as an oracle.”

## What the off condition is

The attempt folder for the off condition contains `prompt.md` and
`data.csv` only. That is not a blank-memory profile. Solvers launched
as agents on this host still inherit the account's standing user rules,
this repository's `AGENTS.md`, and the host skill catalog. The
published contrast is the skill *tree* present versus absent, against
that baseline. A receipt of filenames in the attempt folder does not
record those channels.

## Hygiene

- Agent trees only under gitignored run directories.
- Do not commit per-cell JSON, draws, or machine-local receipts.
- No leftover skill copies or Stan binaries in the working tree.
