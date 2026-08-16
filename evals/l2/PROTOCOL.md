# L2 live protocol

Skill-on vs skill-off, same model, three independent tries, deterministic
grade. This is the live batch. Pytest on fixtures is only the grader check.

## Status (2026-08-16)

**This batch:** Cursor Grok 4.6 only (36 cells). Opus 4.7 / Fable deferred
until Usage events after Grok look fine. Fable is not the default second
model (cost).

**S1 off (Grok 4.6):** pass^1 = 1/3, pass^3 = 0.
**S1 on (Grok 4.6):** pass^1 = 3/3, pass^3 = 1. Delta pass^1 = +2/3.
Scores: `results/S1_without_grok46.json`, `results/S1_with_grok46.json`.

**S2 off (Grok 4.6):** pass^1 = 0/3, pass^3 = 0. All three moved to
non-centered after divergences; all three missed prior predictive, the
R-hat 1.01 threshold, and generated quantities.
Scores: `results/S2_without_grok46.json`.

**S2 on (Grok 4.6):** pass^1 = 3/3, pass^3 = 1. Delta pass^1 = +1.
Scores: `results/S2_with_grok46.json`.

**Running:** S3 skill-off, n = 3, Grok 4.6.

## Design

| Item | Choice |
|---|---|
| Homeworks | S1, S2, S3, S4, S5, S8 |
| Conditions | skill off, then skill on |
| Tries per cell | **n = 3** (independent, blank-memory agents) |
| Primary model | Cursor Grok 4.6 |
| Second model (optional) | Opus 4.7 or Fable — **same** on/off protocol, not vs Grok |
| Prompt | pack `prompt.md` verbatim; no coaching |
| Agent folder | `prompt.md` + `data.csv` only; skill copy only when on |
| Gold | `rubric.json` and pack `meta.json` stay in this repo, never in the folder |
| Grade | Band A (workflow files) + Band B (94% HDI contains hidden truth) |
| Headline numbers | **pass^1** and **pass^3** per model × condition, then the on−off delta |
| Judge | no LLM string-judge; Band C (other model family, pairwise) is later |

pass^k (τ-bench) is the chance that **all** k independent tries succeed,
`C(c,k)/C(n,k)`. pass^1 is the ordinary pass rate. One lucky try is not
enough. SkillsBench averages five tries; three is our floor.

Compare models only to themselves (Grok+skill vs Grok−skill). A weaker or
stronger second model is a second paired delta, not a Grok-vs-Opus contest.

## How a cell is run

1. `python evals/l2/run_trial.py --pack S1 --condition without --n 3`
2. Keep the receipt (file list). Void the cell if gold files appear.
3. Three **new** agents, blank memory, folder = that `rep-*` only, model fixed.
4. `python evals/l2/run_trial.py --pack S1 --condition without --n 3 --grade` (or `grade.py` on each `rep-*`).
5. Copy the small score JSON to `results/` and commit it.
6. `python evals/l2/run_trial.py --wipe --run-root evals/l2/local_runs/<stamp>`

Do not open 36 extra Cursor projects. Do not write `eval_metadata.json`
beside outputs. Do not run solvers in a chat that has seen this protocol.

## Reproducibility

**Fixed (re-runable):** pack CSVs, prompts, grader, rubric, git commit of
this repo. Results JSON records model id, condition, pack, commit SHA, date,
success vector, pass^1, pass^3.

**Not bit-identical:** agent transcripts and MCMC draws. That is why n = 3
and why Band B is “truth inside the HDI,” not “same posterior as the oracle.”

Usage: after the first few cells, check **Usage events** (the per-request
list). The Spending % bar can lag. Cursor-model cells hit the Cursor pool;
Opus/Fable cells hit Other Models.

## Hygiene

- Agent trees only under `evals/l2/local_runs/` (gitignored).
- Wipe after scoring. GitHub keeps `results/*.json` only.
- No leftover skill copies, Stan binaries, or `.cursor` project entries.
