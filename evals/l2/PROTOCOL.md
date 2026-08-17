# L2 live protocol

Paired evaluation of the skill: same homework, same model, skill off then
skill on, three independent attempts, deterministic grade. Fixture pytest
checks the grader only.

## Design

| Item | Choice |
|---|---|
| Homeworks | S1–S8 |
| Conditions | skill off, then skill on |
| Attempts per cell | *n* = 3 (independent, blank-memory agents) |
| Primary model | Cursor Grok 4.6 |
| Optional second model | Opus 4.7 or Fable, same on/off pairing, not vs. Grok |
| Prompt | pack `prompt.md` verbatim; no coaching |
| Agent folder | `prompt.md` and `data.csv` only; skill copy only when on |
| Gold | `rubric.json` and pack `meta.json` stay in this repo |
| Grade | Band A (workflow files) and Band B (94% HDI contains hidden truth) |
| Headline | pass^1 and pass^3 per model × condition, then the on−off delta |
| Judge | no LLM string-judge |

pass^*k* (τ-bench) is the probability that all *k* independent attempts
succeed, *C*(*c*,*k*)/*C*(*n*,*k*). pass^1 is the ordinary pass rate.
SkillsBench averages five attempts; three is the floor used here.

Models are compared only to themselves (Grok with skill vs. Grok without).
A second model family is a second paired delta.

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

## Grok 4.6 results

Date 2026-08-17. Combined pass^1, *n* = 3.

| Pack | Off | On | Δ |
|---|---|---|---|
| S1 | 1/3 | 3/3 | +2/3 |
| S2 | 0/3 | 3/3 | +1 |
| S3 | 1/3 | 3/3 | +2/3 |
| S4 | 3/3 | 3/3 | 0 |
| S5 | 0/3 | 3/3 | +1 |
| S6 | 0/3 | 3/3 | +1 |
| S7 | 2/3 | 3/3 | +1/3 |
| S8 | 0/3 | 3/3 | +1 |

S4, S6, S7, and S8 recovered hidden truth in every scored attempt (off and
on). S1 recovered α, β, and σ in both conditions. S2, S3, and S5 have no
Band B truth.

On S1–S3 and S5, Band A restricted to `report.md` was 5/18 off vs. 18/18
on. The same cut without the `1.01` predicate was 14/18 vs. 18/18.

JSON: `results/S{1–8}_{without,with}_grok46.json`,
`results/grok46_batch.json`, `results/grok46_followup.json`,
`results/grok46_s4.json`.

A second-model arm has not been run.

## Procedure

1. `python evals/l2/run_trial.py --pack S1 --condition without --n 3`
2. Keep the receipt (visible file list).
3. Three new agents, blank memory, working directory is the `rep-*` folder
   only, model fixed.
4. `python evals/l2/run_trial.py --pack S1 --condition without --n 3 --grade`
   (or `grade.py` on each `rep-*`).
5. Write the score JSON with `evals/l2/emit_results.py` and commit
   `results/*.json`.
6. After review, `python evals/l2/run_trial.py --wipe --run-root <run-root>`
   (keeps `report.md` / `.py` / `.stan` / `.json`; drops `.nc`, images,
   binaries).

Do not open extra Cursor projects. Do not write `eval_metadata.json`
beside outputs. Do not run solvers in a chat that has seen this protocol
or the gold. Do not dump library source (`NUTSInfo`, `az.summary`
internals). Bound introspection; time out hung inspect commands.

## Reproducibility

**Fixed:** pack CSVs, prompts, grader, rubric, git commit of this repo.
Results JSON records model id, condition, pack, commit SHA, date, success
vector, pass^1, and pass^3.

**Not bit-identical:** agent transcripts and MCMC draws. That is why
*n* = 3 and why Band B is “truth inside the HDI,” not “same posterior as
an oracle.”

Cursor-model cells use the Cursor usage pool. Other-model cells use
Other Models. Record spend from the per-request event list.

## Hygiene

- Agent trees only under `~/Downloads/jaxs-knife-l2-runs/` (outside the repo).
- GitHub keeps `results/*.json` only. Local `report.md` and other small
  text may remain until wipe.
- No leftover skill copies, Stan binaries, or extra `.cursor` projects.
