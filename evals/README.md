# Tests

```bash
python -m pytest
```

`evals/l0` checks the diagnostic JSON contract on constructed traces (fast).
`evals/smoke` runs live NUTS: CmdStanPy and BlackJAX on the same mini-normal
data. If CmdStan is missing, the smoke installs 2.39.0. JAX and BlackJAX are
required dependencies.

## Inspection gallery (not a gate)

After a live run:

```bash
python evals/smoke/live_nuts.py
```

Open `evals/smoke/artifacts/index.html`. Plots are for looking; ratings come
from `diagnose_model` / `check_diagnostics`. The gallery is gitignored.

## Scenario prompts

`evals/scenarios/` holds prompts only. They are not copied into the skill.
