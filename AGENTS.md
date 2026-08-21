# jaxs-knife

Agent Skill for a Gelman–Vehtari Bayesian workflow on Stan and JAX.
ArviZ `InferenceData` is the shared posterior object.

## Layout

Tracked in a clone:

- `jaxs-knife/` — the skill (`SKILL.md`, `references/`, `scripts/`)
- `evals/` — tests and scenario prompts; not shipped inside the skill folder
- `docs/` — wordmark (name and artwork are not MIT; see `TRADEMARKS.md`)

Gitignored local working directories (not in a clone):

- `.local/` — internal working trees
- `stan_models/` — CmdStan compile cache

A live run uses separate top-level solver and grader directories
outside the clone. Completed run archives may be retained under
`.local/runs/<date>/` (gitignored). Exact launch text is not
published; keep a copy outside the working tree. A gold-free
checklist is `evals/l2/OPERATOR.md`. Do not treat leftover
directories under `.local/` as the next test.

An isolated solver is a fresh agent whose working directory is one
attempt folder (`rep-*`) containing only the files that attempt is
allowed to see. Do not start one from a chat that has seen `evals/`
gold. Do not copy rubric, pack `meta.json`, or eval prompts into
`SKILL.md` or into an agent working directory. Do not open this
repository as a solver workspace: this file is itself a playbook.

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
