# Sealed workflow grades

Pytest checks the grader. The same scripts prepare isolated folders and
score them. Solvers are separate agents; this tree does not launch them.

```bash
python -m pytest evals/l2
```

## Prepare, then grade

```bash
python evals/l2/run_trial.py --pack S1 --condition without --n 3
python evals/l2/run_trial.py --pack S1 --condition with --n 3 --grade
python evals/l2/grade.py --trial /path/to/rep-0
python evals/l2/run_trial.py --wipe --run-root <run-root>
```

Default `--run-root` is repo-root `skill-on-off/` (gitignored). Keep
harvest trees, including draws, unless disk must be reclaimed.
`--wipe` strips `.nc` / images / binaries and is optional.

Design notes: [PROTOCOL.md](PROTOCOL.md).

## Compare two graded batches

```bash
python evals/l2/compare.py --ours ours/batch.json --other other/batch.json --k 1
```

Writes default to `evals/compare/` (gitignored).

## Packs

`packs/S1` … `S8` each ship `prompt.md` and `data.csv`. Prompts do not name
this skill. `meta.json` is not copied into the workspace.
