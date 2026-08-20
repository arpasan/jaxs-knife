# Skill-off vs. skill-on scores

Paired evaluation: same task, skill absent then skill attached, one
Grok 4.6 attempt and one Opus 5 attempt per cell, deterministic
grading. The two attempts in a cell are different models, not two
copies of one model. `on_off.json` is the aggregate published with
the repository. Per-attempt files stay local and are not published.

## Status

`on_off.json` is `not_yet_run`. A prior sealed run was withdrawn
after the instrument-error coverage screen was found to pin
generating latent moments. New scores will be written after a
cue-free rerun on the six sealed tasks.

## What a pass means

An attempt passes only if every workflow-checklist predicate in
[`rubric.json`](../rubric.json) holds — a report with limitations,
probability language, 50% and 94% intervals, a prior predictive check
before sampling, a stated R-hat threshold, draws saved, posterior
predictives from the same likelihood used for inference, positive
scales handled by a constrained declaration, a positive family, or an
explicit Jacobian, and the divergence rule in the rubric (if
divergences exist, refuse to interpret; if none, say so) — and each
recorded generating value falls inside the reported 94% interval.
Those values live in each task's `meta.json` in this repository and
are never copied into an agent's directory.

The checklist is the scientific workflow. It does not require Stan or
JAX.

"Tasks, both solvers" means both models passed that task in that
condition.

Incomplete attempts are graded as they stand.

## Limitations

- Two models, one attempt each per cell. The two scores are not
  independent replicates of one model.
- Agent transcripts and MCMC draws are not bit-reproducible. Fixed
  across runs are the task data, prompts, grader, rubric, and the
  skill tree that was attached.
- Grading is lexical over the agent's files. It rewards stating a
  threshold and reporting an interval; it cannot judge whether the
  prose is insightful.
- Incomplete attempts are graded as they stand and counted as failures.
- The off condition is the skill absent from the host catalog and from
  the attempt folder, after standing user rules were cleared. See
  [`PROTOCOL.md`](../PROTOCOL.md).
