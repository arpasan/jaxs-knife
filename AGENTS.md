# standoff-bayes

Agent Skill for a Gelman–Vehtari Bayesian workflow on Stan and JAX.
ArviZ `InferenceData` is the shared posterior object.

## Layout

- `stan-jax-workflow/` — the skill (`SKILL.md`, `references/`, `scripts/`)
- `evals/` — tests and scenario prompts; not shipped inside the skill folder

## Conventions

- Scripts consume InferenceData. Qualitative ratings come from JSON.
- Public Python functions: full type hints; optional injected `logger`;
  no prints in library code.
- Do not put eval prompts or gold answers in `SKILL.md`.
- L2 gold, `rubric.json`, and pack `meta.json` stay under `evals/`. Never
  copy them into an agent cwd. Do not write `eval_metadata.json` beside
  agent outputs.
- Live L2: same model, skill on vs off, n = 3, report pass^1 and pass^3.
  Primary solver is Cursor Grok 4.6; an optional second stack is Opus 4.7
  or Fable (paired on/off, not vs Grok). See `evals/l2/PROTOCOL.md`.
- After a live batch, commit only `evals/l2/results/*.json`. Wipe
  `local_runs/`. Do not leave extra Cursor projects.
- Do not treat BridgeStan host-callbacks as a JIT/GPU Stan path.
- Do not transpile Stan to XLA.

## Stack

CmdStan 2.39, CmdStanPy 1.3, BridgeStan 2.9 (nutpie may fetch 2.8),
nutpie 0.16.11, BlackJAX 1.6.2, ArviZ current.
