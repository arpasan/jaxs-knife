# Tests

```bash
python -m pytest
```

`evals/l0` checks the diagnostic JSON contract on constructed traces (fast).
`evals/l2` grades sealed fixtures and constructed recovery (fast). The live
skill-on vs skill-off batch is [`l2/PROTOCOL.md`](l2/PROTOCOL.md): same
model, n = 3, pass^1 and pass^3, folders wiped after scoring.
`evals/smoke` runs live NUTS: CmdStanPy and BlackJAX on the same mini-normal
data. If CmdStan is missing, the smoke installs 2.39.0. JAX and BlackJAX are
required dependencies. Live fits must also recover the known DGP inside a
94% HDI.

## Inspection gallery (not a gate)

After a live run:

```bash
python evals/smoke/live_nuts.py
```

Open `evals/smoke/artifacts/index.html`. Plots are for looking; ratings come
from `diagnose_model` / `check_diagnostics`. The gallery is gitignored.

## L2

See [l2/PROTOCOL.md](l2/PROTOCOL.md) and [l2/README.md](l2/README.md).
Agent workspaces are prepared under `evals/l2/local_runs/` (gitignored) and
wiped after scoring. Prompts live in `evals/l2/packs/` and do not name the
skill. `evals/scenarios/` keeps the original prompt stubs.

## Scenario prompts

`evals/scenarios/` holds prompt stubs only. They are not copied into the skill.
