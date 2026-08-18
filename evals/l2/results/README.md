# Skill-on vs. skill-off scores

Same model, n = 3 independent attempts, deterministic grade. Combined
pass is the report and diagnose gates (Band A) and, when a pack records
a hidden data-generating value, whether that value sits in the 94%
interval (Band B). Headline is pass^1 and pass^3.

`on_off.json` is the public score file. It does not name a solver, and
it has no hidden values or per-attempt dumps. Rebuild it from local
per-cell JSON with `python evals/l2/summarize_on_off.py`, passing
`--homework-dir` and `--science-dir` when those files are not under
`skill-on-off/`. `eval_commit` is the skill revision used for the runs,
not the publish commit.

## Homework suite (S1–S8)

Combined pass^1: **5/24 → 21/24**. Band B: **15/15** under both
conditions (five of eight packs record hidden truth). The lift is the
write-up, not recovery of a recoverable DGP.

## Science suite (M1, F1, X1, C1)

Same grader. Four packs: two-component mixture, hierarchical, JAX
location-scale, and a left-truncated logger. Combined pass^1:
**6/12 → 12/12**. Band B: **12/12** under both conditions.

C1 was built so a plain-normal interval cannot cover the process mean.
Skill-off still recovered: the write-threshold in the prompt was
enough. Its combined lift is one Band A miss (50/94 intervals).

Across both suites, attaching the skill moves the report and diagnose
gates. It has not moved whether a 94% interval covers hidden truth.
