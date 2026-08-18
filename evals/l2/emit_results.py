"""Write per-cell score JSON from ``run_trial.py --grade`` output."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from isolation import contamination
from passk import pass_at_k

L2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = L2_ROOT.parents[1]
RESULTS = REPO_ROOT / ".local" / "i-skill-on-off"
DEFAULT_MODEL = "unspecified"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=L2_ROOT.parents[1],
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _band_a_failed(band_a: Dict[str, Any]) -> List[str]:
    predicates = band_a.get("predicates") or []
    return [str(p["id"]) for p in predicates if not p.get("ok")]


def emit_cell(
    *,
    pack: str,
    condition: str,
    model: str,
    git: str,
    grade: Mapping[str, Any],
    without: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the committed per-cell schema from a grade payload.

    Parameters
    ----------
    pack : str
        Pack id.
    condition : str
        ``with`` or ``without``.
    model : str
        Solver model id.
    git : str
        Short commit SHA.
    grade : Mapping[str, Any]
        ``grade_replicates`` / ``batch.json`` ``grade`` object.
    without : Mapping[str, Any], optional
        Off-skill grade, used for on-skill deltas.

    Returns
    -------
    Dict[str, Any]
        Serializable results document.
    """
    trials_in = list(grade.get("trials") or [])
    successes = [bool(x) for x in grade.get("successes") or []]
    if not successes and trials_in:
        successes = [bool(row.get("passed")) for row in trials_in]
    band_a_ok: List[bool] = []
    band_b_ok: List[Optional[bool]] = []
    trials: List[Dict[str, Any]] = []
    for i, row in enumerate(trials_in):
        band_a = row.get("band_a") or {}
        band_b = row.get("band_b")
        a_pass = bool(band_a.get("passed")) if isinstance(band_a, dict) else bool(band_a)
        if band_b is None:
            b_pass: Optional[bool] = None
        elif isinstance(band_b, dict):
            b_pass = bool(band_b.get("passed"))
        else:
            b_pass = bool(band_b)
        band_a_ok.append(a_pass)
        band_b_ok.append(b_pass)
        trial_dir = row.get("trial_dir")
        contam = contamination(Path(trial_dir)) if trial_dir else []
        failed = _band_a_failed(band_a) if isinstance(band_a, dict) else []
        trials.append(
            {
                "rep": i,
                "band_a_failed": failed,
                "band_a": a_pass,
                "band_b": b_pass,
                "band_b_detail": band_b if isinstance(band_b, dict) else None,
                "passed": bool(row.get("passed")),
                "contam": contam,
            }
        )
    payload: Dict[str, Any] = {
        "pack": pack,
        "condition": condition,
        "model": model,
        "git": git,
        "date": date.today().isoformat(),
        "n": len(successes),
        "successes": successes,
        "pass_at_1": pass_at_k(successes, 1) if successes else 0.0,
        "pass_at_3": pass_at_k(successes, 3) if successes else 0.0,
        "band_a_successes": band_a_ok,
        "band_a_pass_at_1": pass_at_k(band_a_ok, 1) if band_a_ok else 0.0,
        "band_b_successes": band_b_ok,
        "band_b_pass_at_1": (
            pass_at_k([bool(x) for x in band_b_ok if x is not None], 1)
            if any(x is not None for x in band_b_ok)
            else None
        ),
        "trials": trials,
    }
    if without is not None and condition == "with":
        off = [bool(x) for x in without.get("successes") or []]
        if off:
            payload["delta_vs_without_pass1"] = payload["pass_at_1"] - pass_at_k(off, 1)
            payload["delta_vs_without_pass3"] = payload["pass_at_3"] - pass_at_k(off, 3)
    return payload


def write_cell(payload: Dict[str, Any], dest: Optional[Path] = None) -> Path:
    """Write ``{pack}_{condition}.json`` under the run directory.

    Parameters
    ----------
    payload : Dict[str, Any]
        Document from ``emit_cell``.
    dest : Path, optional
        Override path.

    Returns
    -------
    Path
        Written file.
    """
    if dest is None:
        cond = "with" if payload["condition"] == "with" else "without"
        dest = RESULTS / f"{payload['pack']}_{cond}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def _load_grade(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "grade" in raw:
        return dict(raw["grade"])
    return dict(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--condition", choices=("with", "without"), required=True)
    parser.add_argument("--batch", type=Path, required=True, help="batch.json from run_trial")
    parser.add_argument("--without-batch", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--git", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    grade = _load_grade(args.batch)
    without = _load_grade(args.without_batch) if args.without_batch else None
    payload = emit_cell(
        pack=args.pack,
        condition=args.condition,
        model=args.model,
        git=args.git or _git_sha(),
        grade=grade,
        without=without,
    )
    dest = write_cell(payload, dest=args.out)
    print(str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
