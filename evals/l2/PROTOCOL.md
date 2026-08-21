# Isolated skill evaluation

Paired evaluation of the skill: same task, skill off then skill on,
two solvers (one attempt each), deterministic grade. The harness lives
under this directory. Live solver trees are not committed.

There is one sealed suite, called **the test**: six tasks, two
conditions, two solvers (24 jobs).

## Design

| Item | Choice |
|---|---|
| Tasks | E1, H1, A1, K1, J1, M1 under `packs/` |
| Conditions | skill off, then skill on |
| Solvers | one Grok 4.6 attempt and one Opus 5 attempt per task and condition |
| Prompt | pack `prompt.md` verbatim; no coaching of the repair |
| Agent folder | `prompt.md` and `data.csv` only |
| Generating values | `meta.json` stays under `evals/l2/packs/`; never copied into the agent folder |
| Grade | workflow checklist on written files; coverage of recorded generating values |
| Headline | attempts passing (out of 12 per condition), tasks where both solvers pass, and coverage |
| Judge | no LLM string-judge |

The two attempts in a cell are different models, not two independent
copies of one model. pass^1 is still the ordinary pass rate over the
twelve attempts in a condition. "Both solvers" is reported separately.
It is not τ-bench pass^3.

## Tasks

| Id | What the prompt states | Engine | Coverage screen |
|---|---|---|---|
| E1 | Response `y`, instrument reading `x`, reported `x_se` | either | Naive OLS misses the slope; a latent-`x` reference that estimates unmarked-`x` moments from the instrument columns covers intercept and slope |
| H1 | `y` grouped by `group`; name `mu`, `tau`, `theta1`, `theta_new` | either | Complete-pool interval misses the group-1 mean and a new-group mean; a hierarchical reference covers `mu`, `tau`, `theta1`, and `theta_new` |
| A1 | Binary assay calls; manufacturer false-positive rate stated | either | Naive proportion misses prevalence; an assay-corrected reference covers it |
| K1 | Sample from a two-component process; name `mu1`, `mu2`, `weight` | either | A single-normal mean misses both locations; a two-component reference covers all three |
| J1 | Positive sample; expert quartiles stated; name `q95` | either | A Gaussian tail interval misses the 95th percentile; a lognormal reference covers it |
| M1 | Response `y` with blank cells; predictor `x`; name `alpha`, `beta` | either | Complete-case OLS misses the slope; a censored-`y` reference covers intercept and slope |

Prompts state the instrument, the grouping, the assay false-positive
rate, stated quartiles, blank cells, or the named estimands. They do
not name an engine (Stan, JAX, BlackJAX) or a repair (`log_mix`,
ordered constraints, Jacobian, truncation syntax, attenuation,
Tobit, JQPD). They do not name this skill.

The workflow checklist is engine-neutral. A PyMC or other density
language that writes the same scientific steps is not failed for the
library used. Coverage is on the named estimands. A1 has no
positive scale, so `constraint_ok` is skipped there (`band_a_skip`).

## Grading

The workflow checklist (implementation: `band_a.py`) is scored on files
the agent wrote. Copied skill trees and `SKILL.md` subtrees are
excluded.

Coverage (implementation: `band_b.py`) is scored on every task. The
grader reads `inference_data.nc` when several NetCDF files are present.
Missing parameter names are recorded as failures. A named estimand
saved as a single plug-in number (or a tiled copy of one number) is
not an interval: Band B fails that name. For `q95`, if the saved
object is a point but `posterior_predictive` has new-unit draws, the
grader forms one 95th percentile per draw and scores that interval.

A cell is discarded if `meta.json`, `rubric.json`, or other gold files
appear in the agent folder: that folder is then no longer a blind
attempt.

## Headline metrics

The published headline is **attempts passing**: the workflow checklist
and coverage of recorded generating values. The two parts are also
stored separately in `results/on_off.json` as `checklist_successes`
and `coverage_successes`.

R-hat is a measured maximum (InferenceData rank-normalized split
R-hat, a diagnostics JSON max, or a stated “max R-hat” in the report).
The predicate is not a search for the characters `1.01`. A robustness
cut that drops the R-hat predicate entirely may still be reported.
Band B is coverage of recorded generating values. Do not revise Band B
after seeing a run.

## Procedure

1. Four sealed solver trees (off-Grok, off-Opus, on-Grok, on-Opus),
   each with six task folders.
2. A fifth tree holds a copy of this harness and writes cell JSON
   with `grade_arms.py`.
3. Rebuild the public aggregate with `evals/l2/summarize_on_off.py`.
   If the grader tree is not a git checkout, pass `--eval-commit` with
   the skill revision used for the run. A complete public file must
   not record `unknown`.
4. Keep live trees under gitignored directories, including draws.

Do not write `eval_metadata.json` beside outputs: a file that names the
condition can contaminate the agent folder and void the cell. Do not
run solvers in a session that has seen this protocol or the generating
values.

Operator steps for a blank-memory host (hide the skill catalog, clear
user rules, restore afterward) live in the gitignored test tree. A
short, gold-free checklist is in [OPERATOR.md](OPERATOR.md).

## Reproducibility

**Fixed:** pack CSVs, prompts, grader, rubric, and the skill revision
used for the run (`eval_commit` in the public file). Local per-cell
JSON records model id, condition, task, commit SHA, date, success
vector, pass^1, and pass^3.

**Not bit-identical:** agent transcripts and MCMC draws. That is why
each cell uses two models, and why coverage is “value inside the
reported interval,” not “same posterior as an oracle.”

## What the off condition is

The attempt folder for the off condition contains `prompt.md` and
`data.csv` only. That is not a blank-memory profile by itself. Solvers
launched as agents on this host still inherit the account's standing
user rules, any `AGENTS.md` in the opened workspace, and the host
skill catalog.

A contrast that can be read as skill-off vs. skill-on therefore
requires, in addition to the folder seal:

- the jaxs-knife clone is **not** the solver workspace (`AGENTS.md` in
  that clone is itself a playbook)
- standing user rules that encode a Stan/JAX workflow are cleared
- the host skill catalog does not expose this skill or another
  Bayesian workflow skill
- the off-arm launch text does not name this skill or tell the solver
  to open `~/.cursor/skills`

The published file states whether those host channels were closed for
the batch. A receipt of filenames in the attempt folder does not
record them.

## Hygiene

- Agent trees only under gitignored run directories.
- Do not commit per-cell JSON, draws, or machine-local receipts.
- No leftover skill copies or Stan binaries in the working tree.
