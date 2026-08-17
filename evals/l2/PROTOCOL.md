# L2 live protocol

Skill-on vs skill-off, same model, three independent tries, deterministic
grade. This is the live batch. Pytest on fixtures is only the grader check.

## Status (2026-08-17)

**This batch:** Cursor Grok 4.6 only. The 18-job follow-up (S6, S7, S8 ×
off/on × n=3) is in. Band B recovered hidden truth **9/9 off and 9/9 on**.
Combined pass^1: S6 0/3→3/3, S7 2/3→3/3, S8 0/3→3/3. That lift is Band A.
Do not claim better posteriors. S4 was re-run on the current CSV with a
Band A `prior_sensitivity_refit` predicate: combined pass^1 **3/3 off and
3/3 on**. Band B recovered α, β, σ on all six tries. Scores:
`results/S{4,6,7,8}_{without,with}_grok46.json`,
`results/grok46_followup.json`, `results/grok46_s4.json`.

Opus 4.7 / Fable deferred until this Grok arm is clean.

**First batch (keep):** S1, S2, S3, S5. Publish Band A with and without
the literal `1.01` predicate. Do not publish raw Band A 18/18 (grader
once read the copied skill tree; fixed). S4/S8 first-batch scores remain
void.

**S1 off (Grok 4.6):** pass^1 = 1/3, pass^3 = 0.
**S1 on (Grok 4.6):** pass^1 = 3/3, pass^3 = 1. Delta pass^1 = +2/3.
Scores: `results/S1_without_grok46.json`, `results/S1_with_grok46.json`.

**S2 off (Grok 4.6):** pass^1 = 0/3, pass^3 = 0. All three moved to
non-centered after divergences; all three missed prior predictive, the
R-hat 1.01 threshold, and generated quantities.
Scores: `results/S2_without_grok46.json`.

**S2 on (Grok 4.6):** pass^1 = 3/3, pass^3 = 1. Delta pass^1 = +1.
Scores: `results/S2_with_grok46.json`.

**S3 off (Grok 4.6):** pass^1 = 1/3, pass^3 = 0. One try missed the
R-hat 1.01 threshold; one missed 50%/94% HDI wording.
Scores: `results/S3_without_grok46.json`.

**S3 on (Grok 4.6):** pass^1 = 3/3, pass^3 = 1. Delta pass^1 = +2/3.
Scores: `results/S3_with_grok46.json`.

**S4 off (Grok 4.6, re-run):** combined pass^1 = 3/3, pass^3 = 1.
Band A 3/3 including `prior_sensitivity_refit`. Band B recovered α, β, σ
on all three. First-batch S4 scores (unrecoverable CSV) stay void and
were overwritten by this cell.
Scores: `results/S4_without_grok46.json`.

**S4 on (Grok 4.6, re-run):** combined pass^1 = 3/3, pass^3 = 1.
Band A 3/3 including the sensitivity refit. Band B 3/3. Combined delta
vs this off arm is 0 after the `inference_data.nc` preference fix
(first grade had off 2/3 from reading `prior_predictive.nc`).
Scores: `results/S4_with_grok46.json`.

**S5 off (Grok 4.6):** pass^1 = 0/3. All three used a constrained `sigma`
and prior predictive; all three missed a limitations section and the
R-hat 1.01 threshold.
Scores: `results/S5_without_grok46.json`.

**S5 on (Grok 4.6):** pass^1 = 3/3, pass^3 = 1. Delta pass^1 = +1.
Scores: `results/S5_with_grok46.json`.

**S8 off (Grok 4.6):** combined pass^1 = 0/3. Band A 0/3 (all missed
prior predictive; two missed R-hat 1.01). Band B 1/3. **Post-critique:**
sample mean was 0.005 ± 0.072 vs truth μ = 0.15 — fixture unrecoverable.
CSV regenerated. Old S8 scores are void. Re-run S8 off+on.
Scores: `results/S8_without_grok46.json` (void).

**Grok 4.6 first batch (36 cells) is in.** Do not publish raw Band A
18/18: the grader read the copied skill tree (fixed in `band_a.py`).
Publish the report.md-only subset: 5/18 off vs 18/18 on, and the same
subset with `rhat_1_01` dropped (14/18 vs 18/18). S1/S2/S3/S5 scores
stay. S4/S8 first-batch scores are void. Aggregate:
`results/grok46_batch.json`. Fable live pass still deferred.

## Design

| Item | Choice |
|---|---|
| Homeworks | S1–S8 (S6 bioassay, S7 mixture added after the first batch) |
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
6. `python evals/l2/emit_results.py --pack S6 --condition without --batch <batch.json>`
7. After review, `python evals/l2/run_trial.py --wipe --run-root <run-root>`
   (keeps `report.md` / `.py` / `.stan` / `.json`; drops `.nc`, images,
   binaries).

Do not open extra Cursor projects. Do not write `eval_metadata.json`
beside outputs. Do not run solvers in a chat that has seen this protocol.
Do not dump library source (`NUTSInfo`, `az.summary` internals). Bound
introspection; time out hung inspect commands.

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

- Agent trees only under `~/Downloads/jaxs-knife-l2-runs/` (outside the repo).
- After scoring and a learning review, wipe heavy artifacts. GitHub keeps
  `results/*.json` only. Small text may stay in the run tree until wipe.
- No leftover skill copies, Stan binaries, or extra `.cursor` projects.
