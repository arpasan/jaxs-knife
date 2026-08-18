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
DEFAULT_CELL_DIR = REPO_ROOT / ".local" / "i-skill-on-off"
STANDARD_IDS: Tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
ADDITIONAL_IDS: Tuple[str, ...] = ("M1", "F1", "X1", "C1")
HOMEWORK_IDS = STANDARD_IDS
SCIENCE_IDS = ADDITIONAL_IDS
TASK_LABELS: Dict[str, str] = {
    "S1": "linear regression",
    "S2": "hierarchical school effects",
    "S3": "overdispersed counts",
    "S4": "linear regression with prior sensitivity",
    "S5": "positive scale (constraint or Jacobian)",
    "S6": "binomial dose-response",
    "S7": "two-component mixture",
    "S8": "JAX log-density",
    "M1": "two-component mixture",
    "F1": "grouped hierarchical",
    "X1": "JAX location-scale",
    "C1": "recordings that omit values below a threshold",
}
SUITE_LABELS: Dict[str, str] = {
    "standard": "Eight reporting tasks",
    "additional": "Four science tasks",
}
COVERAGE_NOTE = (
    "scored only on tasks that record generating values; each such "
    "task's data was accepted only if a reference interval under its "
    "own priors covered them"
)
PASS_DEFINITION = (
    "an attempt passes when every workflow-checklist predicate holds "
    "and, when the task records generating values, the reported 94% "
    "interval contains each of them"
)
REQUIRED_SUITE_KEYS = (
    "label",
    "task_ids",
    "attempt_successes",
    "attempt_pass_at_1",
    "tasks_passing_all_attempts",
    "checklist_successes",
    "coverage_successes",
    "coverage_note",
    "tasks",
)
REQUIRED_TOP_KEYS = (
    "date",
    "eval_commit",
    "attempts_per_cell",
    "pass_definition",
    "standard",
    "additional",
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


def _bool_flags(cell: Mapping[str, Any], *keys: str) -> List[bool]:
    for key in keys:
        raw = cell.get(key)
        if raw is None:
            continue
        flags: List[bool] = []
        for item in raw:
            if item is None:
                continue
            flags.append(bool(item))
        return flags
    return []


def suite_from_cells(
    cells: Mapping[Tuple[str, str], Mapping[str, Any]],
    pack_ids: Sequence[str],
    *,
    suite_key: str,
) -> Dict[str, Any]:
    """Aggregate one suite. Drops solver ids and per-attempt dumps.

    Parameters
    ----------
    cells : Mapping[Tuple[str, str], Mapping[str, Any]]
        Keyed by ``(pack_id, "off"|"on")``.
    pack_ids : Sequence[str]
        Task order.
    suite_key : str
        ``standard`` or ``additional``.

    Returns
    -------
    Dict[str, Any]
        Public suite block.
    """
    rows: List[Dict[str, Any]] = []
    combined: Dict[str, List[bool]] = {"off": [], "on": []}
    checklist: Dict[str, List[bool]] = {"off": [], "on": []}
    coverage: Dict[str, List[bool]] = {"off": [], "on": []}
    all_three: Dict[str, int] = {"off": 0, "on": 0}
    for pack_id in pack_ids:
        row: Dict[str, Any] = {
            "id": pack_id,
            "label": TASK_LABELS[pack_id],
        }
        for cond in ("off", "on"):
            cell = cells[(pack_id, cond)]
            successes = [bool(x) for x in cell.get("successes") or []]
            combined[cond].extend(successes)
            checklist[cond].extend(_bool_flags(cell, "band_a_successes", "checklist_successes"))
            coverage[cond].extend(_bool_flags(cell, "band_b_successes", "coverage_successes"))
            row[f"{cond}_attempts_passing"] = int(sum(successes))
            row[f"{cond}_pass_at_1"] = _rate(float(cell["pass_at_1"]))
            row[f"{cond}_pass_at_3"] = _rate(float(cell["pass_at_3"]))
            if float(cell["pass_at_3"]) == 1.0:
                all_three[cond] += 1
        rows.append(row)
    if len(combined["off"]) != len(combined["on"]):
        raise ValueError("off and on attempt lengths differ")
    if len(coverage["off"]) != len(coverage["on"]):
        raise ValueError("off and on coverage lengths differ")
    return {
        "label": SUITE_LABELS[suite_key],
        "task_ids": list(pack_ids),
        "attempt_successes": {
            "off": int(sum(combined["off"])),
            "on": int(sum(combined["on"])),
            "out_of": int(len(combined["off"])),
        },
        "attempt_pass_at_1": {
            "off": _rate(pass_at_k(combined["off"], 1)) if combined["off"] else 0.0,
            "on": _rate(pass_at_k(combined["on"], 1)) if combined["on"] else 0.0,
        },
        "tasks_passing_all_attempts": {
            "off": int(all_three["off"]),
            "on": int(all_three["on"]),
            "out_of": int(len(pack_ids)),
        },
        "checklist_successes": {
            "off": int(sum(checklist["off"])),
            "on": int(sum(checklist["on"])),
            "out_of": int(len(checklist["off"])),
        },
        "coverage_successes": {
            "off": int(sum(coverage["off"])),
            "on": int(sum(coverage["on"])),
            "out_of": int(len(coverage["off"])),
        },
        "coverage_note": COVERAGE_NOTE,
        "tasks": rows,
    }


def load_suite_cells(
    root: Path,
    pack_ids: Sequence[str],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Read ``{pack}_{with|without}.json`` for each task.

    Parameters
    ----------
    root : Path
        Directory of per-cell score files.
    pack_ids : Sequence[str]
        Task ids.

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
    standard_dir: Path = DEFAULT_CELL_DIR,
    additional_dir: Path = DEFAULT_CELL_DIR,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Assemble the public score document.

    Parameters
    ----------
    standard_dir : Path
        Per-cell JSON for the eight reporting tasks.
    additional_dir : Path
        Per-cell JSON for the four science tasks.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Any]
        Public score document.
    """
    if logger is not None:
        logger.info("reading standard-task cells from %s", standard_dir)
        logger.info("reading additional-task cells from %s", additional_dir)
    standard_cells = load_suite_cells(standard_dir, STANDARD_IDS)
    additional_cells = load_suite_cells(additional_dir, ADDITIONAL_IDS)
    commits = {
        str(cell.get("git") or "")
        for cell in (*standard_cells.values(), *additional_cells.values())
    }
    dates = {
        str(cell.get("date") or "")
        for cell in (*standard_cells.values(), *additional_cells.values())
    }
    commits.discard("")
    dates.discard("")
    if len(commits) != 1:
        raise ValueError(f"mixed or missing eval commits: {sorted(commits)}")
    if len(dates) != 1:
        raise ValueError(f"mixed or missing eval dates: {sorted(dates)}")
    return {
        "date": next(iter(dates)),
        "eval_commit": next(iter(commits)),
        "attempts_per_cell": 3,
        "pass_definition": PASS_DEFINITION,
        "standard": suite_from_cells(standard_cells, STANDARD_IDS, suite_key="standard"),
        "additional": suite_from_cells(
            additional_cells, ADDITIONAL_IDS, suite_key="additional"
        ),
        "notes": [
            "eval_commit is the skill revision used for the runs.",
        ],
    }


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
        Directory of task folders.

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
    n_attempts = int(payload["attempts_per_cell"])
    if n_attempts != 3:
        problems.append("attempts_per_cell must be 3")
    if "Band A" in str(payload.get("pass_definition") or "") or "Band B" in str(
        payload.get("pass_definition") or ""
    ):
        problems.append("pass_definition must not use Band A / Band B")
    for suite_name in ("standard", "additional"):
        suite = payload[suite_name]
        if not isinstance(suite, dict):
            problems.append(f"{suite_name} is not an object")
            continue
        for key in REQUIRED_SUITE_KEYS:
            if key not in suite:
                problems.append(f"{suite_name} missing {key}")
        task_ids = list(suite.get("task_ids") or [])
        expected = list(STANDARD_IDS if suite_name == "standard" else ADDITIONAL_IDS)
        if task_ids != expected:
            problems.append(f"{suite_name} task_ids {task_ids} != {expected}")
        for task_id in task_ids:
            if not (packs_root / task_id / "prompt.md").is_file():
                problems.append(f"missing tracked task {task_id}")
            if not (packs_root / task_id / "data.csv").is_file():
                problems.append(f"missing data.csv for {task_id}")
        attempts = suite.get("attempt_successes") or {}
        out_of = int(attempts.get("out_of") or 0)
        if out_of != n_attempts * len(task_ids):
            problems.append(f"{suite_name} attempt out_of {out_of} is inconsistent")
        for cond in ("off", "on"):
            successes = int(attempts.get(cond) or 0)
            if not 0 <= successes <= out_of:
                problems.append(f"{suite_name} {cond} successes out of range")
            expected_rate = _rate(successes / out_of) if out_of else 0.0
            reported = _rate(
                float((suite.get("attempt_pass_at_1") or {}).get(cond) or 0.0)
            )
            if expected_rate != reported:
                problems.append(
                    f"{suite_name} {cond} pass^1 {reported} != {expected_rate}"
                )
        rows = list(suite.get("tasks") or [])
        row_ids = [str(row.get("id")) for row in rows]
        if row_ids != task_ids:
            problems.append(f"{suite_name} task rows {row_ids} != {task_ids}")
        for row in rows:
            if not row.get("label"):
                problems.append(f"{row.get('id')} missing label")
            for key in (
                "off_attempts_passing",
                "on_attempts_passing",
                "off_pass_at_1",
                "on_pass_at_1",
                "off_pass_at_3",
                "on_pass_at_3",
            ):
                if key not in row:
                    problems.append(f"{row.get('id')} missing {key}")
    if "model" in payload or "solver" in payload:
        problems.append("public file must not contain a solver field")
    return problems


def per_cell_trees_available(
    standard_dir: Path = DEFAULT_CELL_DIR,
    additional_dir: Path = DEFAULT_CELL_DIR,
) -> bool:
    """True when both local per-cell trees are present."""
    try:
        load_suite_cells(standard_dir, STANDARD_IDS)
        load_suite_cells(additional_dir, ADDITIONAL_IDS)
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
        "--standard-dir",
        "--homework-dir",
        dest="standard_dir",
        type=Path,
        default=DEFAULT_CELL_DIR,
        help="directory of standard-task per-cell JSON",
    )
    parser.add_argument(
        "--additional-dir",
        "--science-dir",
        dest="additional_dir",
        type=Path,
        default=DEFAULT_CELL_DIR,
        help="directory of additional-task per-cell JSON",
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
        standard_dir=args.standard_dir,
        additional_dir=args.additional_dir,
    )
    problems = validate_public_on_off(payload)
    if problems:
        raise SystemExit("\n".join(problems))
    dest = write_public_on_off(payload, dest=args.out)
    print(str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
