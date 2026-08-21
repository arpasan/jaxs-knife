"""Grade four flat solver trees (one attempt per task per arm)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from emit_results import _git_sha, emit_cell
from grade import grade_trial
from summarize_on_off import TASK_IDS
from workspace import pack_aliases, pack_band_a_extra, pack_band_a_skip, pack_truth


def grade_flat_arm(arm_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Grade ``arm_dir/{pack}/`` for each sealed task.

    Parameters
    ----------
    arm_dir : Path
        Directory that contains one folder per sealed task.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Pack id → ``grade_trial`` report.
    """
    root = arm_dir.resolve()
    out: Dict[str, Dict[str, Any]] = {}
    for pack_id in TASK_IDS:
        trial = root / pack_id
        if not trial.is_dir():
            raise FileNotFoundError(f"missing task folder {trial}")
        out[pack_id] = grade_trial(
            trial,
            truth=pack_truth(pack_id),
            aliases=pack_aliases(pack_id),
            extra_band_a=pack_band_a_extra(pack_id),
            skip_band_a=pack_band_a_skip(pack_id),
        )
    return out


def _join_condition(
    pack_id: str,
    condition: str,
    grok: Mapping[str, Any],
    opus: Mapping[str, Any],
    git: str,
) -> Dict[str, Any]:
    """Stack Grok then Opus into one per-cell document."""
    grade = {
        "successes": [bool(grok.get("passed")), bool(opus.get("passed"))],
        "trials": [dict(grok), dict(opus)],
    }
    payload = emit_cell(
        pack=pack_id,
        condition=condition,
        model="grok-4.6+opus-5",
        git=git,
        grade=grade,
    )
    payload["models"] = ["grok-4.6", "opus-5"]
    payload["date"] = date.today().isoformat()
    return payload


def write_cells(
    arms: Mapping[str, Mapping[str, Mapping[str, Any]]],
    dest: Path,
    *,
    git: Optional[str] = None,
) -> Dict[str, Path]:
    """Write ``{pack}_{without|with}.json`` for the summarizer.

    Parameters
    ----------
    arms : Mapping[str, Mapping[str, Mapping[str, Any]]]
        Arm name → pack id → grade.
    dest : Path
        Output directory.
    git : str, optional
        Short SHA. Read from the repo if omitted.

    Returns
    -------
    Dict[str, Path]
        Cell id → written path.
    """
    dest.mkdir(parents=True, exist_ok=True)
    sha = git or _git_sha()
    written: Dict[str, Path] = {}
    for pack_id in TASK_IDS:
        for cond, grok_key, opus_key in (
            ("without", "off-grok", "off-opus"),
            ("with", "on-grok", "on-opus"),
        ):
            payload = _join_condition(
                pack_id,
                cond,
                arms[grok_key][pack_id],
                arms[opus_key][pack_id],
                sha,
            )
            path = dest / f"{pack_id}_{cond}.json"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            written[f"{pack_id}_{cond}"] = path
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-grok", type=Path, required=True)
    parser.add_argument("--off-opus", type=Path, required=True)
    parser.add_argument("--on-grok", type=Path, required=True)
    parser.add_argument("--on-opus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--git",
        default=None,
        help="skill revision SHA; do not rely on git rev-parse from an archived tree",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    named = {
        "off-grok": args.off_grok,
        "off-opus": args.off_opus,
        "on-grok": args.on_grok,
        "on-opus": args.on_opus,
    }
    arms: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, path in named.items():
        arms[key] = grade_flat_arm(path)
    cell_dir = args.out / "cells"
    write_cells(arms, cell_dir, git=args.git)
    (args.out / "arms.json").write_text(
        json.dumps({k: {p: arms[k][p]["passed"] for p in TASK_IDS} for k in named}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
