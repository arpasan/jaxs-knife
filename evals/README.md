# Tests

```bash
python -m pytest
```

`evals/l0` checks the diagnostic JSON contract on constructed traces (fast).
`evals/l2` grades isolated fixtures and constructed recovery (fast).
`evals/smoke` runs live NUTS: CmdStanPy and BlackJAX on the same mini-normal
data. If CmdStan is missing, the smoke installs 2.39.0. JAX and BlackJAX are
required dependencies. Live fits must also recover the known generating
values inside a 94% HDI.

`evals/l1` scores how example queries overlap the `SKILL.md` description.
It is a manual diagnostic, not part of `pytest`:

```bash
python evals/l1/score_triggers.py
```

## Inspection gallery (not a gate)

After a live smoke run:

```bash
python evals/smoke/live_nuts.py
```

Open `evals/smoke/artifacts/index.html`. Plots are for inspection; ratings
are computed by `diagnose_model` / `check_diagnostics`. The gallery is
gitignored.

## Tasks and prompts

`evals/l2/packs/` holds the six isolated tasks (E1, H1, A1, K1, J1, M1).
Each folder contains `prompt.md` and `data.csv`. Generating values live
in `meta.json` and are not copied into an agent workspace. The packs
are not copied into the skill.
