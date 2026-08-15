# L2 — sealed workflow grades

This script **is** the test. Pytest checks the grader. Live skill-on vs
skill-off runs use the same script to prepare folders and to score them.
Solvers are separate blank-memory agents. After a batch, a small JSON
summary is copied to `results/` (committed). Agent folders are wiped.

```bash
python -m pytest evals/l2
```

## Prepare, then grade (the used path)

```bash
python evals/l2/run_trial.py --pack S1 --condition without --n 3
python evals/l2/run_trial.py --pack S1 --condition with --n 3 --grade
python evals/l2/grade.py --trial /path/to/rep-0
python evals/l2/run_trial.py --wipe --run-root evals/l2/local_runs/<stamp>
```

`local_runs/` is gitignored. Wipe deletes `rep-*` trees and keeps receipts
and `batch.json`. Do not create extra Cursor projects for trials.

## Compare two graded batches

```bash
python evals/l2/compare.py --ours ours/batch.json --other other/batch.json --k 1
```

Writes default to `evals/compare/` (gitignored).

## Packs

`packs/S1` … `S8` each ship `prompt.md` and `data.csv`. Prompts do not name
this skill. `meta.json` is not copied into the workspace.
