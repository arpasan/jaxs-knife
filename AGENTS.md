# jaxs-knife

Agent Skill for a Gelman–Vehtari Bayesian workflow on Stan and JAX.
ArviZ `InferenceData` is the shared posterior object.

## Layout

- `jaxs-knife/` — the skill (`SKILL.md`, `references/`, `scripts/`)
- `evals/` — tests and scenario prompts; not shipped inside the skill folder
- `docs/` — wordmark (name and artwork are not MIT; see `TRADEMARKS.md`)
- `skill-on-off/` — gitignored sealed harvest (keep draws and reports)
- `.local/` — gitignored internal trees (rehearsals, notes, engine compares)
- `stan_models/` — gitignored CmdStan compile cache; empty on purpose

Do not start sealed solvers from a chat that has seen `evals/` gold.
Do not copy rubric, pack `meta.json`, or eval prompts into `SKILL.md`
or into an agent working directory.

## Conventions

- Scripts consume InferenceData. Qualitative ratings come from JSON.
- Public Python functions: full type hints; optional injected `logger`;
  no prints in library code.
- Do not put eval prompts or gold answers in `SKILL.md`.
- Rubric and pack metadata stay under `evals/`. Never copy them into an
  agent working directory.
- Do not treat BridgeStan host-callbacks as a JIT/GPU Stan path.
- Do not transpile Stan to XLA.

## Stack

CmdStan 2.39, CmdStanPy 1.3, BridgeStan 2.9 (nutpie may fetch 2.8),
nutpie 0.16.11, BlackJAX 1.6.2, ArviZ current.
