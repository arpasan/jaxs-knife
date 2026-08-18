# jaxs-knife

Agent Skill for a Gelman–Vehtari Bayesian workflow on Stan and JAX.
ArviZ `InferenceData` is the shared posterior object.

## Layout

Tracked in a clone:

- `jaxs-knife/` — the skill (`SKILL.md`, `references/`, `scripts/`)
- `evals/` — tests and scenario prompts; not shipped inside the skill folder
- `docs/` — wordmark (name and artwork are not MIT; see `TRADEMARKS.md`)

Gitignored local working directories (not in a clone):

- `.local/` — internal working trees, including the numbered test trees
- `stan_models/` — CmdStan compile cache

Numbered test trees live directly under `.local/`. A roman-numeral
prefix marks a test tree; anything in `.local/` without one is shared
tooling or research material, not evidence.

- `.local/i-skill-on-off/` — reporting suite, skill off vs. on
- `.local/ii-engine-compare/` — unpublished engine compare
- `.local/iii-science-on-off/` — science suite, skill off vs. on
- `.local/iv-rehearsals/` — author demonstrations, not an evaluation

Each local working directory carries its own `README.md` stating what it
holds, what was held fixed, how to read it, and what must not be
touched. Read that file before running anything inside one, and before
changing a published number that depends on it. The run trees are the
only evidence behind `evals/l2/results/on_off.json`, and they are not
reproducible bit-for-bit if deleted. `.local/README.md` indexes the rest.
Nothing from these directories — model ids, comparator names,
per-attempt content, or generating values in transit — belongs in a
tracked file.

An isolated solver is a fresh agent whose working directory is one
attempt folder (`rep-*`) containing only the files that attempt is
allowed to see. Do not start one from a chat that has seen `evals/`
gold. Do not copy rubric, pack `meta.json`, or eval prompts into
`SKILL.md` or into an agent working directory.

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
