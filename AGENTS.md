# jaxs-knife

Agent Skill for a Gelman–Vehtari Bayesian workflow on Stan and JAX.
ArviZ `InferenceData` is the shared posterior object.

## Layout

- `jaxs-knife/` — the skill (`SKILL.md`, `references/`, `scripts/`)
- `evals/` — tests and scenario prompts; not shipped inside the skill folder
- `docs/` — wordmark (name and artwork are not MIT; see `TRADEMARKS.md`)

Local working directories (not in a clone): `.local/`, `stan_models/`.

Do not copy rubric, pack `meta.json`, or eval prompts into `SKILL.md`
or into an agent working directory. Do not open this tree as a solver
workspace: this file is itself a playbook.

## Conventions

- Scripts consume InferenceData. Qualitative ratings come from JSON.
- Public Python functions: full type hints; optional injected `logger`;
  no prints in library code.
- Do not treat BridgeStan host-callbacks as a JIT/GPU Stan path.
- Do not transpile Stan to XLA.

## Stack

CmdStan 2.39, CmdStanPy 1.3, BridgeStan 2.9 (nutpie may fetch 2.8),
nutpie 0.16.11, BlackJAX 1.6.2, ArviZ current.
