"""Prepare isolated trial folders and grade them from outside.

This script does not launch solvers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from grade import grade_trial
from isolation import IsolationError
from workspace import (
    Condition,
    pack_aliases,
    pack_band_a_extra,
    pack_truth,
    prepare_workspace,
)

L2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = L2_ROOT.parents[1]
DEFAULT_RUNS = REPO_ROOT / ".local" / "test"
KEEP_AFTER_WIPE = frozenset({"batch.json", "receipt.json", "grade.json"})
KEEP_TEXT_SUFFIXES = frozenset({".md", ".py", ".stan", ".json", ".txt"})
DROP_ON_WIPE = frozenset({".nc", ".png", ".pdf", ".npy", ".so", ".o", ".hpp", ".d"})


def new_run_root(base: Path = DEFAULT_RUNS) -> Path:
    """Create a timestamped run directory under ``base``.

    Parameters
    ----------
    base : Path
        Parent for live run trees (gitignored).

    Returns
    -------
    Path
        New empty directory.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = base / stamp
    dest.mkdir(parents=True, exist_ok=False)
    return dest


def prepare_replicates(
    pack_id: str,
    *,
    n: int,
    condition: Condition,
    run_root: Path,
) -> List[Path]:
    """Prepare ``n`` isolated workspaces for one pack and condition.

    Parameters
    ----------
    pack_id : str
        Scenario pack.
    n : int
        Replicate count.
    condition : Condition
        Skill on or off.
    run_root : Path
        Parent directory for this batch.

    Returns
    -------
    List[Path]
        Workspace paths.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    paths: List[Path] = []
    for i in range(n):
        dest = run_root / pack_id / condition / f"rep-{i}"
        prepare_workspace(pack_id, dest, condition=condition)
        paths.append(dest)
    return paths


def write_receipt(workspace: Path) -> Path:
    """Write the agent-visible file list. Used to prove no gold leaked in.

    Parameters
    ----------
    workspace : Path
        Prepared trial directory.

    Returns
    -------
    Path
        ``receipt.json`` next to the workspace (parent), not inside it.
    """
    names = sorted(
        p.relative_to(workspace).as_posix()
        for p in workspace.rglob("*")
        if p.is_file()
    )
    payload = {
        "workspace": str(workspace.resolve()),
        "files": names,
        "sealed": True,
    }
    out = workspace.parent / f"{workspace.name}.receipt.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def grade_replicates(workspaces: List[Path], pack_id: str) -> Dict[str, Any]:
    """Grade each workspace from outside the agent cwd.

    Parameters
    ----------
    workspaces : List[Path]
        Trial directories.
    pack_id : str
        Pack id (truth is read from pack meta, never from the workspace).

    Returns
    -------
    Dict[str, Any]
        Per-trial grades and a success vector.
    """
    truth = pack_truth(pack_id)
    aliases = pack_aliases(pack_id)
    extra = pack_band_a_extra(pack_id)
    rows: List[Dict[str, Any]] = []
    successes: List[bool] = []
    for ws in workspaces:
        report = grade_trial(
            ws, truth=truth, aliases=aliases, extra_band_a=extra
        )
        rows.append(report)
        successes.append(bool(report["passed"]))
    return {
        "pack_id": pack_id,
        "n": len(workspaces),
        "successes": successes,
        "trials": rows,
    }


def wipe_workspaces(run_root: Path, *, keep_text: bool = True) -> int:
    """Strip heavy artifacts under ``run_root``. Keep grade/receipt JSON.

    When ``keep_text`` is true, retain ``report.md`` and other small text
    (``.py``, ``.stan``, ``.json``) so a later regrade is possible. Drop
    ``.nc``, images, and compiled Stan bits.

    Parameters
    ----------
    run_root : Path
        A timestamped run directory.
    keep_text : bool
        Keep small text artifacts inside ``rep-*`` folders.

    Returns
    -------
    int
        Number of workspace directories removed (full wipe) or stripped.
    """
    root = run_root.resolve()
    removed = 0
    if not root.exists():
        return 0
    for path in sorted(root.rglob("rep-*"), reverse=True):
        if not (path.is_dir() and path.name.startswith("rep-")):
            continue
        if not keep_text:
            shutil.rmtree(path)
            removed += 1
            continue
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            drop = suffix in DROP_ON_WIPE or (
                suffix not in KEEP_TEXT_SUFFIXES and file_path.stat().st_size > 200_000
            )
            if drop:
                file_path.unlink()
        removed += 1
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default=None)
    parser.add_argument("--condition", choices=("with", "without"), default=None)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--grade",
        action="store_true",
        help="grade prepared workspaces after setup",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="delete agent folders under --run-root; keep JSON receipts/grades",
    )
    args = parser.parse_args(argv)

    if args.wipe:
        if args.run_root is None:
            print("--wipe requires --run-root", file=sys.stderr)
            return 2
        n = wipe_workspaces(args.run_root)
        print(json.dumps({"wiped_workspaces": n, "run_root": str(args.run_root)}))
        return 0

    if args.pack is None or args.condition is None:
        print("--pack and --condition are required unless --wipe", file=sys.stderr)
        return 2

    run_root = args.run_root or new_run_root()
    try:
        workspaces = prepare_replicates(
            args.pack,
            n=args.n,
            condition=args.condition,
            run_root=run_root,
        )
        receipts = [str(write_receipt(ws)) for ws in workspaces]
    except IsolationError as exc:
        print(f"isolation error: {exc}", file=sys.stderr)
        return 2

    payload: Dict[str, Any] = {
        "run_root": str(run_root.resolve()),
        "pack": args.pack,
        "condition": args.condition,
        "workspaces": [str(p) for p in workspaces],
        "receipts": receipts,
    }
    if args.grade:
        payload["grade"] = grade_replicates(workspaces, args.pack)

    out = run_root / args.pack / args.condition / "batch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
