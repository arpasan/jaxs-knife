# Sealed skill evaluation

Paired evaluation of the skill: same homework, same model, skill off then
skill on, three independent attempts, deterministic grade. The harness
lives under this directory. Live solver trees are not committed.

## Design

| Item | Choice |
|---|---|
| Homeworks | S1–S8 |
| Conditions | skill off, then skill on |
| Attempts per cell | *n* = 3 (independent agents, no shared memory) |
| Model | record the model id; compare a model only to itself |
| Prompt | pack `prompt.md` verbatim; no coaching |
| Agent folder | `prompt.md` and `data.csv` only; skill copy only when on |
| Gold | `rubric.json` and pack `meta.json` stay in this repo |
| Grade | Band A (workflow files) and Band B (94% HDI contains hidden truth) |
| Headline | pass^1 and pass^3 per model × condition, then the on−off delta |
| Judge | no LLM string-judge |

pass^*k* (τ-bench) is the probability that all *k* independent attempts
succeed, *C*(*c*,*k*)/*C*(*n*,*k*). pass^1 is the ordinary pass rate.

S4 and S8 CSVs are generated so a reference 94% interval under the pack
priors covers the hidden truth. S6 is a binomial dose–response. S7 is a
two-component mixture.

## Grading

Band A is scored on files the agent wrote. Copied skill trees and
`SKILL.md` subtrees are excluded. S4 adds an optional
`prior_sensitivity_refit` predicate from pack `meta.json`.

Band B is scored when the pack records hidden truth. The grader reads
`inference_data.nc` when several NetCDF files are present. Missing
parameter names are recorded as failures.

A cell is discarded if gold files appear in the agent folder.

## Headline metrics

The primary headline is **combined** pass (Band A, and Band B when that
pack is scored). Band A and Band B are also reported separately. Band B
measures whether hidden truth lies in the 94% HDI; it is not a claim that
the skill produces a better posterior.

A robustness cut reports Band A with and without the literal `1.01`
R-hat predicate.

## Procedure

1. `python evals/l2/run_trial.py --pack S1 --condition without --n 3`
2. Keep the receipt (visible file list).
3. Three new agents, no shared memory, working directory is the `rep-*`
   folder only, model fixed.
4. `python evals/l2/run_trial.py --pack S1 --condition without --n 3 --grade`
   (or `grade.py` on each `rep-*`).
5. Write the score JSON with `evals/l2/emit_results.py` into a gitignored
   run directory. Do not commit per-cell JSON.
6. After review, `python evals/l2/run_trial.py --wipe --run-root <run-root>`
   (keeps `report.md` / `.py` / `.stan` / `.json`; drops `.nc`, images,
   binaries).

Do not write `eval_metadata.json` beside outputs. Do not run solvers in a
session that has seen this protocol or the gold. Do not dump library
source (`NUTSInfo`, `az.summary` internals). Bound introspection; time
out hung inspect commands.

## Reproducibility

**Fixed:** pack CSVs, prompts, grader, rubric, git commit of this repo.
Score JSON records model id, condition, pack, commit SHA, date, success
vector, pass^1, and pass^3.

**Not bit-identical:** agent transcripts and MCMC draws. That is why
*n* = 3 and why Band B is “truth inside the HDI,” not “same posterior as
an oracle.”

## Hygiene

- Agent trees only under repo-root `skill-on-off/` (gitignored).
- Do not commit per-cell JSON, draws, or machine-local receipts.
- No leftover skill copies or Stan binaries in the working tree.
