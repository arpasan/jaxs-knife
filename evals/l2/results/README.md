# Skill-off vs. skill-on scores

Paired evaluation: same task, same model, skill absent then skill
attached, three independent attempts per cell, deterministic grading.
`on_off.json` is the aggregate published with the repository.
Per-attempt files stay local and are not published.

## What a pass means

An attempt passes only if every workflow-checklist predicate in
[`rubric.json`](../rubric.json) holds — a report with limitations,
probability language, 50% and 94% intervals, a prior predictive check
before sampling, a stated R-hat threshold, draws saved, predictive
checks in `generated quantities` or `vmap`, positive scales handled by
a constrained declaration or an explicit Jacobian, and the divergence
rule in the rubric (if divergences exist, refuse to interpret; if none,
say so) — and, when the task records the parameter values used to
generate its data, each value falls inside the reported 94% interval.
Those values live in each task's `meta.json` in this repository and are
never copied into an agent's directory.

"Tasks passing 3 of 3" means every attempt on that task passed.

Incomplete attempts are graded as they stand.

## Results

The first suite is eight reporting exercises. The second is four
science problems (mixture, hierarchical, JAX location-scale, recordings
that drop low values), scored with the same write-up checklist.

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

Attempts passing, per task, out of three:

```text
+------+-----------------------------------------------+-----------+----------+
| Task | What the prompt asks                          | skill off | skill on |
+------+-----------------------------------------------+-----------+----------+
| S1   | linear regression                             |     2 / 3 |    3 / 3 |
| S2   | hierarchical school effects                   |     1 / 3 |    1 / 3 |
| S3   | overdispersed counts                          |     0 / 3 |    3 / 3 |
| S4   | linear regression with prior sensitivity      |     1 / 3 |    3 / 3 |
| S5   | positive scale (constraint or Jacobian)       |     0 / 3 |    3 / 3 |
| S6   | binomial dose-response                        |     1 / 3 |    3 / 3 |
| S7   | two-component mixture                         |     0 / 3 |    3 / 3 |
| S8   | JAX log-density                               |     0 / 3 |    2 / 3 |
+------+-----------------------------------------------+-----------+----------+
| M1   | two-component mixture                         |     3 / 3 |    3 / 3 |
| F1   | grouped hierarchical                          |     0 / 3 |    3 / 3 |
| X1   | JAX location-scale                            |     1 / 3 |    3 / 3 |
| C1   | recordings that omit values below a threshold |     2 / 3 |    3 / 3 |
+------+-----------------------------------------------+-----------+----------+
```

## Reading the result

The difference between conditions is entirely in the workflow checklist.
Coverage of the recorded generating values was complete in both
conditions. Each scored task's data was accepted only if a reference
interval under the task's own priors covered the value, so coverage is
a floor check on the fit, not a test that separates the conditions.

With the skill attached, hierarchical school effects still passes only
one attempt in three, and the JAX log-density task passes two in three.
The skill raises the rate at which a complete, auditable write-up is
produced; it does not guarantee one.

## Limitations

- One model, three attempts per cell. The aggregate gap is large; the
  per-task fractions are coarse.
- Agent transcripts and MCMC draws are not bit-reproducible. Fixed
  across runs are the task data, prompts, grader, rubric, and skill
  revision.
- Grading is lexical over the agent's files. It rewards stating a
  threshold and reporting an interval; it cannot judge whether the
  prose is insightful.
- Incomplete attempts are graded as they stand and counted as failures.
