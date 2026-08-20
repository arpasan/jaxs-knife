# Operator checklist (gold-free)

Solvers open a sealed folder, not this repository. Do not start a
solver from a chat that has opened `evals/`.

Pack prompts name the scientific story (instrument SE, grouping, assay
FPR, two-component sample, stated quartiles, blank cells). They do
not name an engine or a repair. The workflow checklist grades those
steps; it does not require Stan or JAX.

## Sequence

1. Clear Cursor Settings → User Rules. Quit the app fully and reopen.
2. Host Bayesian skills stay hidden until an on-solver puts **only**
   this skill shortcut back.
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

Launch text for the five windows lives in the gitignored test tree
(`.local/test/LAUNCH_PROMPTS.md` when that tree is present).
