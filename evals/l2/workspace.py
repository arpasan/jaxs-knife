"""Build an isolated agent workspace from a scenario pack."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping

from isolation import FORBIDDEN_NAMES, IsolationError, assert_sealed

L2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = L2_ROOT.parents[1]
PACKS = L2_ROOT / "packs"
SKILL_SRC = REPO_ROOT / "jaxs-knife"
SKILL_NAME = "jaxs-knife"

Condition = Literal["with", "without"]


def load_pack(pack_id: str) -> Dict[str, Any]:
    """Load pack metadata. ``meta.json`` is never copied to the agent.

    Parameters
    ----------
    pack_id : str
        Directory name under ``packs/``.

    Returns
    -------
    Dict[str, Any]
        Parsed metadata plus ``pack_dir``.
    """
    pack_dir = (PACKS / pack_id).resolve()
    meta_path = pack_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"unknown pack: {pack_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"invalid meta.json in {pack_id}")
    meta["pack_dir"] = pack_dir
    return meta


def prepare_workspace(
    pack_id: str,
    dest: Path,
    *,
    condition: Condition = "without",
) -> Path:
    """Copy only agent-visible files into ``dest``.

    Parameters
    ----------
    pack_id : str
        Pack id (e.g. ``E1``).
    dest : Path
        Empty directory that will become the agent cwd.
    condition : Condition
        ``with`` copies the skill folder into ``.cursor/skills/``.
        ``without`` copies prompt and data only.

    Returns
    -------
    Path
        Resolved ``dest``.
    """
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise IsolationError(f"destination is not empty: {dest}")

    meta = load_pack(pack_id)
    pack_dir: Path = meta["pack_dir"]
    names = list(meta.get("copy_to_agent") or [])
    if not names:
        raise ValueError(f"{pack_id}: copy_to_agent is empty")
    for name in names:
        if Path(name).name in FORBIDDEN_NAMES:
            raise IsolationError(f"{pack_id}: refuses to copy {name}")
        src = pack_dir / name
        if not src.is_file():
            raise FileNotFoundError(src)
        target = dest / Path(name).name
        shutil.copy2(src, target)

    if condition == "with":
        skill_dest = dest / ".cursor" / "skills" / SKILL_NAME
        shutil.copytree(
            SKILL_SRC,
            skill_dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    assert_sealed(dest)
    _assert_no_evals_leak(dest)
    return dest


def _assert_no_evals_leak(dest: Path) -> None:
    """The agent cwd must not contain the evals tree or this grader."""
    leaked = []
    for rel in ("evals", "evals/l2", "evals/l2/rubric.json"):
        if (dest / rel).exists():
            leaked.append(rel)
    if leaked:
        raise IsolationError("evals tree leaked into workspace: " + ", ".join(leaked))


def pack_truth(pack_id: str) -> Mapping[str, float] | None:
    """Return synthetic truth for Band B, or None if the pack has none.

    Parameters
    ----------
    pack_id : str
        Pack id.

    Returns
    -------
    Mapping[str, float] | None
        Parameter name → true value.
    """
    meta = load_pack(pack_id)
    truth = meta.get("truth")
    if truth is None:
        return None
    return {str(k): float(v) for k, v in dict(truth).items()}


def pack_band_a_extra(pack_id: str) -> List[str]:
    """Optional extra Band A predicate ids from pack ``meta.json``."""
    meta = load_pack(pack_id)
    raw = meta.get("band_a_extra") or []
    return [str(item) for item in raw]


def pack_aliases(pack_id: str) -> Dict[str, List[str]]:
    """Optional Band B name aliases from pack ``meta.json``.

    Parameters
    ----------
    pack_id : str
        Pack id.

    Returns
    -------
    Dict[str, List[str]]
        Canonical name → accepted posterior names.
    """
    meta = load_pack(pack_id)
    raw = meta.get("aliases") or {}
    return {str(k): [str(item) for item in v] for k, v in dict(raw).items()}
