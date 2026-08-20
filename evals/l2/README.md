# Isolated workflow grades

Pytest checks the grader. The same scripts prepare isolated folders and
score them. This tree does not launch agents.

```bash
python -m pytest evals/l2
```

## Prepare, then grade

```bash
python evals/l2/run_trial.py --pack E1 --condition without --n 3
python evals/l2/run_trial.py --pack E1 --condition with --n 3 --grade
python evals/l2/grade.py --trial /path/to/rep-0
python evals/l2/run_trial.py --wipe --run-root <run-root>
```

Default `--run-root` is `.local/test/` (gitignored). Keep those run
trees, including draws, unless disk must be reclaimed. `--wipe` strips
`.nc` / images / binaries and is optional.

Design notes: [PROTOCOL.md](PROTOCOL.md). Host isolation:
[OPERATOR.md](OPERATOR.md).

## Compare two graded batches

```bash
python evals/l2/compare.py --ours ours/batch.json --other other/batch.json --k 1
```

Writes default to `evals/compare/` (gitignored).

## Tasks

Each folder under `packs/` ships `prompt.md` and `data.csv`. `meta.json`
is not copied into the workspace. Public aggregate scores:
[results/on_off.json](results/on_off.json).
