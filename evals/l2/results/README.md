# Skill-absent vs. skill-attached scores

Paired evaluation: same task, skill absent then skill attached, one
Grok 4.6 attempt and one Opus 5 attempt per task, deterministic
grading. The two attempts on a task are different models, not two
copies of one model. `on_off.json` is the aggregate published with
the repository. Per-attempt files stay local and are not published.

## Status

`on_off.json` is `complete` for the 21 August 2026 run at skill
revision `f718e5c882e7dffb645ea7d68daac1ed637417e8`.

```
Metric                         Skill absent   Skill attached   Denominator
-----------------------------  ------------   --------------   -----------
Attempts passing                          9               12            12
  component: workflow checklist          11               12            12
  component: recorded-value coverage      9               12            12
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

The checklist line is a component of the same attempts, not a second
independent result. The combined 9 / 12 vs. 12 / 12 is the coverage
delta plus one skill-absent attempt that also failed the measured
R-hat threshold.

One skill-absent attempt on the instrument-error task failed the
measured R-hat threshold and missed the slope. Both skill-absent
attempts on the blank-response task treated blanks as
missing-at-random and missed the slope; both skill-attached attempts
used a censored likelihood on the blanks and covered the named
estimands. The other four tasks passed in both conditions.

A 20 August 2026 run was withdrawn before publication because a
coverage screen encoded latent reference moments; its scores are not
reported.

## What a pass means

An attempt passes only if every workflow-checklist predicate in
[`rubric.json`](../rubric.json) holds — a report with limitations,
probability language, 50% and 94% intervals, a prior predictive check
before sampling, a measured R-hat maximum at or below 1.01, draws
saved, posterior predictives from the same likelihood used for
inference, positive scales handled by a constrained declaration, a
positive family, or an explicit Jacobian, and the divergence rule in
the rubric (if divergences exist, refuse to interpret; if none, say
so) — and each recorded generating value falls inside the reported
94% interval. Those values live in each task's `meta.json` in this
repository and are never copied into an agent's directory.

R-hat is read from the saved draws, a diagnostics JSON maximum, or a
stated maximum in the report. The predicate is not a search for the
characters `1.01`. Coverage is numerical over the reported interval.

The checklist is the scientific workflow. It does not require Stan or
JAX.

"Tasks, both solvers" means both models passed that task in that
condition.

Incomplete attempts are graded as they stand.

## Limitations

- Two models, one attempt each per task and condition. The two scores
  are not independent replicates of one model.
- Conditions were run in a fixed order, not randomized.
- Three of twelve paired attempts differed, all in the same direction
  (exact McNemar two-sided *p* = 0.25). This is a description of one
  run, not a measured effect size.
- Skill-absent attempts used PyMC; skill-attached attempts used Stan
  via CmdStanPy. The contrast does not isolate skill text from the
  implementation path it induced.
- The observation-model guidance exercised by the blank-response task
  was written in this same revision, after earlier attempts on that
  task failed. The run checks that the guidance is followed; it is
  not an out-of-sample estimate of skill effect.
- Hierarchical pooling, assay correction, mixtures, and elicited
  lognormal tails already passed without the skill on this suite. The
  skill did not invent those methods here.
- Agent transcripts and MCMC draws are not bit-reproducible. Fixed
  across runs are the task data, prompts, grader, rubric, and the
  skill tree that was attached.
- The grader tree was not a git checkout. `eval_commit` is the skill
  revision supplied at publication.
- Incomplete attempts are graded as they stand and counted as
  failures.
- The skill-absent condition is the skill absent from the host
  catalog and from the attempt folder, after standing user rules were
  cleared. See [`PROTOCOL.md`](../PROTOCOL.md).
