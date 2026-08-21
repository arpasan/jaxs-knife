# Skill-absent vs. skill-attached scores

Same six tasks, skill attached vs. left off, one Grok 4.6 attempt and
one Opus 5 attempt each. A script grades the files the agent wrote.
`on_off.json` is the published aggregate. Per-attempt files are not.

## Scores

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

**Checklist** — report, 50% and 94% intervals, prior predictive check,
diagnostics (measured R-hat, not the characters `1.01`), limitations,
draws on disk. Engine-neutral.

**Coverage** — each recorded generating value in the reported 94%
interval. Values live in `evals/l2/packs/*/meta.json` and are never
copied into an agent's directory.

The 9 → 12 gain is coverage: both skill-absent blank-response attempts
used ignorable missingness and missed the slope; both skill-attached
attempts used a censored likelihood and covered. One skill-absent
instrument-error attempt also failed measured R-hat and missed the
slope. Hierarchical pooling, assay correction, mixtures, and elicited
tails already passed without the skill.

One run, two models, not two copies of one model. Design:
[PROTOCOL.md](../PROTOCOL.md).
