# S8 — JAX constrain + BlackJAX → ArviZ (JAX required)

Prompt only. No gold answers. Do not copy into SKILL.md.

Write a JAX `logdensity_fn` with an explicit constrain + Jacobian, sample with
BlackJAX NUTS, convert draws to ArviZ, and run the diagnose → check_diagnostics
scripts. Do not transpile Stan to JAX.
