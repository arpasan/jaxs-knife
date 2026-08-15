"""Score two sealed result trees with the same portable rubric.

Pass ``--other-dir`` (and optionally ``--other-skill-dir`` for bookkeeping).
Do not hard-code another skill's name. Do not add API-locked predicates.
Writes under ``--out`` (default: evals/compare/, gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from passk import pass_at_k, skill_delta


def _success_vector(batch_path: Path) -> List[bool]:
    payload = json.loads(batch_path.read_text(encoding="utf-8"))
    grade = payload.get("grade") or {}
    raw = grade.get("successes")
    if raw is None:
        raise ValueError(f"{batch_path}: missing grade.successes; re-run with --grade")
    return [bool(x) for x in raw]


def compare_batches(
    ours: Path,
    other: Path,
    *,
    k: int,
) -> Dict[str, Any]:
    """Compute pass^k and the skill delta.

    Parameters
    ----------
    ours, other : Path
        ``batch.json`` files produced by ``run_trial.py --grade``.
    k : int
        pass^k order.

    Returns
    -------
    Dict[str, Any]
        Portable comparison payload.
    """
    a = _success_vector(ours)
    b = _success_vector(other)
    return {
        "k": k,
        "ours": {"n": len(a), "successes": a, "pass_at_k": pass_at_k(a, k)},
        "other": {"n": len(b), "successes": b, "pass_at_k": pass_at_k(b, k)},
        "delta_ours_minus_other": skill_delta(a, b, k),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, required=True, help="our batch.json")
    parser.add_argument("--other", type=Path, required=True, help="other batch.json")
    parser.add_argument(
        "--other-skill-dir",
        type=Path,
        default=None,
        help="optional path to the other skill (bookkeeping only; never imported)",
    )
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "compare" / "latest.json",
    )
    args = parser.parse_args(argv)

    report = compare_batches(args.ours, args.other, k=args.k)
    if args.other_skill_dir is not None:
        report["other_skill_dir"] = str(args.other_skill_dir.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
