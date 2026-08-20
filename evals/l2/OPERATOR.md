# Operator checklist (gold-free)

Solvers open a sealed folder, not this repository. Do not start a
solver from a chat that has opened `evals/`.

Pack prompts name the scientific story (instrument SE, grouping, assay
FPR, two-component sample, stated quartiles, blank cells). They do
not name an engine or a repair. All six share the same workflow
footer. The workflow checklist grades those steps; it does not
require Stan or JAX. R-hat is a measured maximum on the saved
draws (or diagnostics JSON / a stated max), not the characters
`1.01`. Coverage (Band B) is unchanged: recorded values inside
the reported 94% interval.

## Sequence

1. Clear Cursor Settings → User Rules. Quit the app fully and reopen.
2. Host Bayesian skills stay hidden **outside** `~/.cursor/skills`
   (sibling folder `~/.cursor/_hidden_for_eval`) until an on-solver
   puts **only** this skill folder back. That folder must be a copy
   of the skill, not a symlink into this repository. The grader tree
   is a separate folder, not a sibling of the four arms.
3. Four solver windows, in order: off Grok, off Opus, on Grok, on Opus.
   Pick that model in the Cursor model menu. File → Open Folder on that
   sealed tree. New chat. Paste the prompt for that tree.
4. Fifth window: the grader folder (this harness copy). New Grok 4.6
   chat. That agent grades; it does not restore the machine.
5. Paste User Rules back from the file you copied them out of.
6. Reopen this repository and restore the remaining host skills, check
   the grade, and write the public tables.

The skill folder that Cursor attaches is the inner skill directory, not
the repository root. Do not clone the public repository as a solver
workspace.

Launch paste-blocks for the five windows live in the operator chat,
not in this repository and not under `.local/test/`.
