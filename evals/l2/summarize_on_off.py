"""Build the public on/off score file from local per-cell JSON."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from passk import pass_at_k

L2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = L2_ROOT.parents[1]
PUBLIC_PATH = L2_ROOT / "results" / "on_off.json"
DEFAULT_CELL_DIR = REPO_ROOT / "skill-on-off"
HOMEWORK_IDS: Tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
SCIENCE_IDS: Tuple[str, ...] = ("M1", "F1", "X1", "C1")
REQUIRED_SUITE_KEYS = (
    "pack_ids",
    "combined_successes",
    "combined_pass_at_1",
    "band_b_successes",
    "packs",
)
REQUIRED_TOP_KEYS = (
    "date",
    "eval_commit",
    "n",
    "headline",
    "homework",
    "science",
)


def _rate(value: float) -> float:
    return float(round(float(value), 4))


def _load_cell(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a JSON object")
    return dict(raw)


def _cell_path(root: Path, pack_id: str, condition: str) -> Path:
    suffix = "with" if condition == "on" else "without"
    return root / f"{pack_id}_{suffix}.json"


def _band_b_flags(cell: Mapping[str, Any]) -> List[bool]:
    flags: List[bool] = []
    for item in cell.get("band_b_successes") or []:
        if item is None:
            continue
        flags.append(bool(item))
    return flags


def suite_from_cells(
    cells: Mapping[Tuple[str, str], Mapping[str, Any]],
    pack_ids: Sequence[str],
) -> Dict[str, Any]:
    """Aggregate one suite. Drops solver ids and per-attempt dumps.

    Parameters
    ----------
    cells : Mapping[Tuple[str, str], Mapping[str, Any]]
        Keyed by ``(pack_id, "off"|"on")``.
    pack_ids : Sequence[str]
        Pack order.

    Returns
    -------
    Dict[str, Any]
        Public suite block.
    """
    rows: List[Dict[str, Any]] = []
    combined: Dict[str, List[bool]] = {"off": [], "on": []}
    band_b: Dict[str, List[bool]] = {"off": [], "on": []}
    for pack_id in pack_ids:
        row: Dict[str, Any] = {"id": pack_id}
        for cond in ("off", "on"):
            cell = cells[(pack_id, cond)]
            successes = [bool(x) for x in cell.get("successes") or []]
            combined[cond].extend(successes)
            band_b[cond].extend(_band_b_flags(cell))
            row[f"{cond}_pass_at_1"] = _rate(float(cell["pass_at_1"]))
            row[f"{cond}_pass_at_3"] = _rate(float(cell["pass_at_3"]))
        rows.append(row)
    combined_successes = {
        "off": int(sum(combined["off"])),
        "on": int(sum(combined["on"])),
        "out_of": int(len(combined["off"])),
    }
    if len(combined["off"]) != len(combined["on"]):
        raise ValueError("off and on combined lengths differ")
    band_b_successes = {
        "off": int(sum(band_b["off"])),
        "on": int(sum(band_b["on"])),
        "out_of": int(len(band_b["off"])),
    }
    if len(band_b["off"]) != len(band_b["on"]):
        raise ValueError("off and on Band B lengths differ")
    return {
        "pack_ids": list(pack_ids),
        "combined_successes": combined_successes,
        "combined_pass_at_1": {
            "off": _rate(pass_at_k(combined["off"], 1)) if combined["off"] else 0.0,
            "on": _rate(pass_at_k(combined["on"], 1)) if combined["on"] else 0.0,
        },
        "band_b_successes": band_b_successes,
        "packs": rows,
    }


def load_suite_cells(
    root: Path,
    pack_ids: Sequence[str],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Read ``{pack}_{with|without}.json`` for each pack.

    Parameters
    ----------
    root : Path
        Directory of per-cell score files.
    pack_ids : Sequence[str]
        Pack ids.

    Returns
    -------
    Dict[Tuple[str, str], Dict[str, Any]]
        Keyed by ``(pack_id, "off"|"on")``.
    """
    cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
    missing: List[str] = []
    for pack_id in pack_ids:
        for cond in ("off", "on"):
            path = _cell_path(root, pack_id, cond)
            if not path.is_file():
                missing.append(str(path))
                continue
            cells[(pack_id, cond)] = _load_cell(path)
    if missing:
        raise FileNotFoundError("missing per-cell files:\n" + "\n".join(missing))
    return cells


def build_public_on_off(
    *,
    homework_dir: Path = DEFAULT_CELL_DIR,
    science_dir: Path = DEFAULT_CELL_DIR,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Assemble the public score document. Omits solver ids.

    Parameters
    ----------
    homework_dir : Path
        Per-cell JSON for S1–S8 (gitignored run tree).
    science_dir : Path
        Per-cell JSON for the science suite.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Any]
        Public score document.
    """
    if logger is not None:
        logger.info("reading homework cells from %s", homework_dir)
        logger.info("reading science cells from %s", science_dir)
    homework_cells = load_suite_cells(homework_dir, HOMEWORK_IDS)
    science_cells = load_suite_cells(science_dir, SCIENCE_IDS)
    commits = {
        str(cell.get("git") or "")
        for cell in (*homework_cells.values(), *science_cells.values())
    }
    dates = {
        str(cell.get("date") or "")
        for cell in (*homework_cells.values(), *science_cells.values())
    }
    commits.discard("")
    dates.discard("")
    if len(commits) != 1:
        raise ValueError(f"mixed or missing eval commits: {sorted(commits)}")
    if len(dates) != 1:
        raise ValueError(f"mixed or missing eval dates: {sorted(dates)}")
    eval_commit = next(iter(commits))
    date = next(iter(dates))
    payload: Dict[str, Any] = {
        "date": date,
        "eval_commit": eval_commit,
        "n": 3,
        "headline": (
            "combined pass is Band A, and Band B when a pack records hidden truth"
        ),
        "homework": suite_from_cells(homework_cells, HOMEWORK_IDS),
        "science": suite_from_cells(science_cells, SCIENCE_IDS),
        "notes": [
            "eval_commit is the skill revision used for the runs.",
            "This file omits solver ids, per-attempt dumps, and hidden values.",
        ],
    }
    return payload


def write_public_on_off(
    payload: Mapping[str, Any],
    dest: Path = PUBLIC_PATH,
    *,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Write ``on_off.json``.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Document from ``build_public_on_off``.
    dest : Path
        Output path.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Path
        Written file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    if logger is not None:
        logger.info("wrote %s", dest)
    return dest


def validate_public_on_off(
    payload: Mapping[str, Any],
    *,
    packs_root: Path = L2_ROOT / "packs",
) -> List[str]:
    """Return human-readable problems, or an empty list if the file is sound.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Public score document.
    packs_root : Path
        Directory of pack folders.

    Returns
    -------
    List[str]
        Problems. Empty means valid.
    """
    problems: List[str] = []
    for key in REQUIRED_TOP_KEYS:
        if key not in payload:
            problems.append(f"missing top-level key {key}")
    if problems:
        return problems
    if int(payload["n"]) != 3:
        problems.append("n must be 3")
    for suite_name in ("homework", "science"):
        suite = payload[suite_name]
        if not isinstance(suite, dict):
            problems.append(f"{suite_name} is not an object")
            continue
        for key in REQUIRED_SUITE_KEYS:
            if key not in suite:
                problems.append(f"{suite_name} missing {key}")
        pack_ids = list(suite.get("pack_ids") or [])
        expected = list(HOMEWORK_IDS if suite_name == "homework" else SCIENCE_IDS)
        if pack_ids != expected:
            problems.append(f"{suite_name} pack_ids {pack_ids} != {expected}")
        for pack_id in pack_ids:
            if not (packs_root / pack_id / "prompt.md").is_file():
                problems.append(f"missing tracked pack {pack_id}")
            if not (packs_root / pack_id / "data.csv").is_file():
                problems.append(f"missing data.csv for {pack_id}")
        combined = suite.get("combined_successes") or {}
        out_of = int(combined.get("out_of") or 0)
        if out_of != int(payload["n"]) * len(pack_ids):
            problems.append(f"{suite_name} combined out_of {out_of} is inconsistent")
        for cond in ("off", "on"):
            successes = int(combined.get(cond) or 0)
            if not 0 <= successes <= out_of:
                problems.append(f"{suite_name} {cond} successes out of range")
            expected_rate = _rate(successes / out_of) if out_of else 0.0
            reported = _rate(float((suite.get("combined_pass_at_1") or {}).get(cond) or 0.0))
            if expected_rate != reported:
                problems.append(
                    f"{suite_name} {cond} pass^1 {reported} != {expected_rate}"
                )
        rows = list(suite.get("packs") or [])
        row_ids = [str(row.get("id")) for row in rows]
        if row_ids != pack_ids:
            problems.append(f"{suite_name} pack rows {row_ids} != {pack_ids}")
        for row in rows:
            for key in (
                "off_pass_at_1",
                "on_pass_at_1",
                "off_pass_at_3",
                "on_pass_at_3",
            ):
                if key not in row:
                    problems.append(f"{row.get('id')} missing {key}")
    if "model" in payload or "solver" in payload:
        problems.append("public file must not name a solver")
    return problems


def per_cell_trees_available(
    homework_dir: Path = DEFAULT_CELL_DIR,
    science_dir: Path = DEFAULT_CELL_DIR,
) -> bool:
    """True when both local per-cell trees are present."""
    try:
        load_suite_cells(homework_dir, HOMEWORK_IDS)
        load_suite_cells(science_dir, SCIENCE_IDS)
    except FileNotFoundError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed public file only",
    )
    parser.add_argument(
        "--homework-dir",
        type=Path,
        default=DEFAULT_CELL_DIR,
        help="directory of S1–S8 per-cell JSON",
    )
    parser.add_argument(
        "--science-dir",
        type=Path,
        default=DEFAULT_CELL_DIR,
        help="directory of science-suite per-cell JSON",
    )
    parser.add_argument("--out", type=Path, default=PUBLIC_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.check:
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        problems = validate_public_on_off(payload)
        if problems:
            raise SystemExit("\n".join(problems))
        print(str(args.out))
        return 0
    payload = build_public_on_off(
        homework_dir=args.homework_dir,
        science_dir=args.science_dir,
    )
    problems = validate_public_on_off(payload)
    if problems:
        raise SystemExit("\n".join(problems))
    dest = write_public_on_off(payload, dest=args.out)
    print(str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
