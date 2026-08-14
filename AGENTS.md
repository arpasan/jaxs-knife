# standoff-bayes

Private bakeoff: one Agent Skill that runs a Gelman–Vehtari workflow on
**Stan** (CmdStanPy / nutpie[stan]) and **JAX** (log-density + BlackJAX).
ArviZ `InferenceData` is the shared posterior object.

This is not a fork of baygent-skills. Diagnostic thresholds and the
`report.md` shape are adapted from baygent’s Bayesian workflow (MIT) and
retargeted. Do not clone amortized-workflow.

## Layout

- `stan-jax-workflow/` — the skill (`SKILL.md`, `references/`, `scripts/`)
- `evals/` — sealed from the skill. Gold traces and assertions stay here.
- `literature/` — local PDFs; gitignored (copyright)

## Conventions

- One skill folder. Split Stan vs. JAX only if L1 shows a confusion pattern
  (one engine systematically misses, or broadening the description spikes
  false positives on PyMC / causal / SBI prompts). No 0.9 P/R cutoff.
- Scripts consume InferenceData. Ratings come from JSON, never vibes.
- Public Python functions: full type hints; optional injected `logger`;
  no prints in library code.
- Do not put eval prompts or gold answers in `SKILL.md`.
- Do not treat BridgeStan host-callbacks as a JIT/GPU Stan path.
- Do not transpile Stan to XLA.

## Stack (v0 pin)

CmdStan 2.39, CmdStanPy 1.3, BridgeStan 2.9 (nutpie may fetch 2.8),
nutpie 0.16.11, BlackJAX 1.6.2, ArviZ current.
