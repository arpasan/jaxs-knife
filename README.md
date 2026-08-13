# standoff-bayes

Private bakeoff for an opinionated **Stan and JAX** Bayesian Agent Skill
(CmdStanPy / nutpie[stan] / BlackJAX), workflow-first, ArviZ as the shared
posterior object.

This is not a fork of [baygent-skills](https://github.com/Learning-Bayesian-Statistics/baygent-skills).
v0 does not clone amortized-workflow.

## Status

Research constitution is complete (science, stack, eval layers). Skill files
(`SKILL.md`, `references/`, `scripts/`) are not written yet. Implementation
starts after the plan is approved.

Working title `standoff-bayes` is fine for this private repo.

## v0 intent

- **Stan stays.** JAX is a peer engine, not a replacement.
- Same Gelman–Vehtari sequence on both engines.
- nutpie[stan] is faster NUTS on a Stan model when BridgeStan is available;
  otherwise CmdStanPy NUTS.
- JAX path: write the log-density like Stan (constrain, Jacobian, `vmap`
  generated quantities) + BlackJAX. Do not transpile Stan to XLA.
- Do not treat BridgeStan host-callbacks as a JIT/GPU Stan path.

## Local notes

`literature/` may hold local PDFs. They are gitignored (copyright).
