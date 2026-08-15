"""Seal a trial directory: gold and rubric stay outside the agent cwd."""

from __future__ import annotations

from pathlib import Path
from typing import FrozenSet, List

FORBIDDEN_NAMES: FrozenSet[str] = frozenset(
    {
        "rubric.json",
        "eval_metadata.json",
        "truth.json",
        "gold.json",
        "meta.json",
    }
)


class IsolationError(ValueError):
    """Trial directory is contaminated with evaluator artifacts."""


def contamination(trial_dir: Path) -> List[str]:
    """Return relative paths of forbidden files under ``trial_dir``.

    Parameters
    ----------
    trial_dir : Path
        Agent workspace.

    Returns
    -------
    List[str]
        Relative POSIX paths that must not be visible to the agent.
    """
    root = trial_dir.resolve()
    hits: List[str] = []
    if not root.exists():
        return hits
    for path in root.rglob("*"):
        if path.is_file() and path.name in FORBIDDEN_NAMES:
            hits.append(path.relative_to(root).as_posix())
    return hits


def assert_sealed(trial_dir: Path) -> None:
    """Raise ``IsolationError`` if evaluator artifacts leaked into the trial.

    Parameters
    ----------
    trial_dir : Path
        Agent workspace.

    Returns
    -------
    None
    """
    hits = contamination(trial_dir)
    if hits:
        raise IsolationError(
            "evaluator artifacts in trial directory: " + ", ".join(hits)
        )
