"""Grade a sealed trial directory from outside the agent cwd."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from band_a import evaluate_band_a
from band_b import assess_recovery, posterior_from_idata
from isolation import IsolationError, assert_sealed


def find_inference_data(trial_dir: Path) -> Optional[Path]:
    """Prefer ``inference_data.nc`` over prior-predictive or sensitivity files."""
    root = trial_dir.resolve()
    preferred = sorted(root.rglob("inference_data.nc"))
    if preferred:
        return preferred[0]
    found = sorted(root.rglob("*.nc"))
    return found[0] if found else None


def grade_trial(
    trial_dir: Path,
    *,
    truth: Optional[Mapping[str, float]] = None,
    aliases: Optional[Mapping[str, Sequence[str]]] = None,
    extra_band_a: Optional[Sequence[str]] = None,
    idata_path: Optional[Path] = None,
    nominal: float = 0.94,
) -> Dict[str, Any]:
    """Run Band A, and Band B when truth and draws are available.

    Parameters
    ----------
    trial_dir : Path
        Agent output directory.
    truth : Mapping[str, float] | None
        Known parameter values. Must come from outside ``trial_dir``.
    idata_path : Path | None
        InferenceData netcdf. Defaults to the first ``*.nc`` under the trial.
    nominal : float
        HDI probability for Band B.

    Returns
    -------
    Dict[str, Any]
        Combined grade. ``passed`` is Band A (and Band B when scored).
    """
    root = trial_dir.resolve()
    assert_sealed(root)
    band_a = evaluate_band_a(root, extra=extra_band_a)

    band_b: Dict[str, Any] | None = None
    if truth:
        nc = idata_path if idata_path is not None else find_inference_data(root)
        if nc is None:
            band_b = {
                "scored": False,
                "passed": False,
                "error": "no InferenceData for Band B",
            }
        else:
            import arviz as az

            try:
                idata = az.from_netcdf(nc)
                posterior = posterior_from_idata(
                    idata, tuple(truth.keys()), aliases=aliases
                )
                band_b = assess_recovery(posterior, truth, nominal=nominal)
                band_b["scored"] = True
            except Exception as exc:
                band_b = {
                    "scored": True,
                    "passed": False,
                    "error": f"Band B extract failed: {exc}",
                }

    passed = bool(band_a["passed"])
    if band_b is not None:
        passed = passed and bool(band_b.get("passed"))

    return {
        "trial_dir": str(root),
        "band_a": band_a,
        "band_b": band_b,
        "passed": passed,
    }


def _parse_truth(raw: str | None) -> Optional[Dict[str, float]]:
    if not raw:
        return None
    payload = json.loads(raw)
    return {str(k): float(v) for k, v in dict(payload).items()}


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Evaluator process; do not run this inside the agent cwd."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument(
        "--truth",
        type=str,
        default=None,
        help="JSON object of true parameter values (evaluator-side only)",
    )
    parser.add_argument("--truth-file", type=Path, default=None)
    parser.add_argument("--idata", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    truth = _parse_truth(args.truth)
    if args.truth_file is not None:
        truth = {
            str(k): float(v)
            for k, v in json.loads(args.truth_file.read_text(encoding="utf-8")).items()
        }
        if args.truth_file.resolve().is_relative_to(args.trial.resolve()):
            raise IsolationError("--truth-file must live outside the trial directory")

    try:
        report = grade_trial(args.trial, truth=truth, idata_path=args.idata)
    except IsolationError as exc:
        print(f"isolation error: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
