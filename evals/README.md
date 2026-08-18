# Tests

```bash
python -m pytest
```

`evals/l0` checks the diagnostic JSON contract on constructed traces (fast).
`evals/l2` grades sealed fixtures and constructed recovery (fast).
`evals/smoke` runs live NUTS: CmdStanPy and BlackJAX on the same mini-normal
data. If CmdStan is missing, the smoke installs 2.39.0. JAX and BlackJAX are
required dependencies. Live fits must also recover the known DGP inside a
94% HDI.

## Inspection gallery (not a gate)

After a live smoke run:

```bash
python evals/smoke/live_nuts.py
```

Open `evals/smoke/artifacts/index.html`. Plots are for looking; ratings come
from `diagnose_model` / `check_diagnostics`. The gallery is gitignored.

## Packs and prompts

`evals/l2/packs/` holds sealed homework packs. Prompts do not name the skill.
`evals/scenarios/` holds prompt stubs only. Neither is copied into the skill.
