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
DEFAULT_CELL_DIR = REPO_ROOT / ".local" / "test"
TASK_IDS: Tuple[str, ...] = ("E1", "H1", "A1", "K1", "J1", "M1")
TASK_LABELS: Dict[str, str] = {
    "E1": "predictor with a reported instrument error",
    "H1": "grouped observations and a new group",
    "A1": "assay with a stated false-positive rate",
    "K1": "two-component sample",
    "J1": "positive sample with stated quartiles",
    "M1": "predictor with some blank responses",
}
SUITE_LABEL = "Six isolated benchmark tasks"
MODELS: Tuple[str, ...] = ("grok-4.6", "opus-5")
ATTEMPTS_PER_CELL = 2
COVERAGE_NOTE = (
    "scored on every task; each CSV is accepted only when a naive "
    "interval misses the named estimand and a reference interval under "
    "the task's observation model covers it"
)
PASS_DEFINITION = (
    "an attempt passes when every workflow-checklist predicate holds "
    "and the reported 94% interval contains each recorded generating value"
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
REQUIRED_COMPLETE_KEYS = (
    "status",
    "date",
    "eval_commit",
    "attempts_per_cell",
    "pass_definition",
    "task_ids",
    *REQUIRED_SUITE_KEYS,
)
REQUIRED_PENDING_KEYS = (
    "status",
    "attempts_per_cell",
    "task_ids",
    "pass_definition",
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


def _eval_commit_ok(value: str) -> bool:
    """True when ``value`` looks like a git SHA, not a placeholder."""
    text = str(value or "").strip().lower()
    if text in {"", "unknown", "none"}:
        return False
    return bool(len(text) >= 7 and len(text) <= 40 and all(
        ch in "0123456789abcdef" for ch in text
    ))


def _commit_matches(supplied: str, known: Sequence[str]) -> bool:
    """True when ``supplied`` is the same revision as a known cell SHA."""
    left = str(supplied or "").strip().lower()
    if not _eval_commit_ok(left):
        return False
    for raw in known:
        right = str(raw or "").strip().lower()
        if not _eval_commit_ok(right):
            continue
        if left == right or left.startswith(right) or right.startswith(left):
            return True
    return False


def pending_public_on_off() -> Dict[str, Any]:
    """Public document used before a skill-absent / skill-attached batch has been graded."""
    return {
        "status": "not_yet_run",
        "attempts_per_cell": ATTEMPTS_PER_CELL,
        "models": list(MODELS),
        "task_ids": list(TASK_IDS),
        "pass_definition": PASS_DEFINITION,
        "notes": [
            "Scores will be written after a skill-absent / skill-attached run.",
            "Each cell is one Grok 4.6 attempt and one Opus 5 attempt, not two copies of one model.",
            "The suite is six tasks (24 jobs).",
        ],
    }


def suite_from_cells(
    cells: Mapping[Tuple[str, str], Mapping[str, Any]],
    pack_ids: Sequence[str],
) -> Dict[str, Any]:
    """Aggregate the isolated suite. Drops solver ids and per-attempt dumps.

    Parameters
    ----------
    cells : Mapping[Tuple[str, str], Mapping[str, Any]]
        Keyed by ``(pack_id, "off"|"on")``.
    pack_ids : Sequence[str]
        Task order.

    Returns
    -------
    Dict[str, Any]
        Public suite fields (merged into the top-level document).
    """
    rows: List[Dict[str, Any]] = []
    combined: Dict[str, List[bool]] = {"off": [], "on": []}
    checklist: Dict[str, List[bool]] = {"off": [], "on": []}
    coverage: Dict[str, List[bool]] = {"off": [], "on": []}
    both: Dict[str, int] = {"off": 0, "on": 0}
    for pack_id in pack_ids:
        row: Dict[str, Any] = {
            "id": pack_id,
            "label": TASK_LABELS[pack_id],
        }
        for cond in ("off", "on"):
            cell = cells[(pack_id, cond)]
            successes = [bool(x) for x in cell.get("successes") or []]
            combined[cond].extend(successes)
            checklist[cond].extend(
                _bool_flags(cell, "band_a_successes", "checklist_successes")
            )
            coverage[cond].extend(
                _bool_flags(cell, "band_b_successes", "coverage_successes")
            )
            row[f"{cond}_attempts_passing"] = int(sum(successes))
            row[f"{cond}_pass_at_1"] = _rate(float(cell["pass_at_1"]))
            row[f"{cond}_both_models"] = int(bool(successes) and all(successes))
            if row[f"{cond}_both_models"]:
                both[cond] += 1
        rows.append(row)
    if len(combined["off"]) != len(combined["on"]):
        raise ValueError("off and on attempt lengths differ")
    if len(coverage["off"]) != len(coverage["on"]):
        raise ValueError("off and on coverage lengths differ")
    return {
        "label": SUITE_LABEL,
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
            "off": int(both["off"]),
            "on": int(both["on"]),
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
    cell_dir: Path = DEFAULT_CELL_DIR,
    eval_commit: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Assemble the public score document from a completed batch.

    Parameters
    ----------
    cell_dir : Path
        Per-cell JSON for the six isolated tasks.
    eval_commit : str, optional
        Skill revision to record. Required when the grader tree was not
        a git checkout and the cells say ``unknown``. Refused when it
        conflicts with a known cell SHA.
    logger : logging.Logger, optional
        Injected logger.

    Returns
    -------
    Dict[str, Any]
        Public score document with ``status`` ``complete``.
    """
    if logger is not None:
        logger.info("reading cells from %s", cell_dir)
    cells = load_suite_cells(cell_dir, TASK_IDS)
    commits = {str(cell.get("git") or "") for cell in cells.values()}
    dates = {str(cell.get("date") or "") for cell in cells.values()}
    commits.discard("")
    dates.discard("")
    if len(commits) != 1:
        raise ValueError(f"mixed or missing eval commits: {sorted(commits)}")
    if len(dates) != 1:
        raise ValueError(f"mixed or missing eval dates: {sorted(dates)}")
    derived = next(iter(commits))
    if eval_commit is not None:
        known = [c for c in commits if _eval_commit_ok(c)]
        if known and not _commit_matches(eval_commit, known):
            raise ValueError(
                f"eval_commit {eval_commit} conflicts with cell git {sorted(known)}"
            )
        sha = str(eval_commit).strip()
    else:
        sha = derived
    payload = {
        "status": "complete",
        "date": next(iter(dates)),
        "eval_commit": sha,
        "attempts_per_cell": ATTEMPTS_PER_CELL,
        "models": list(MODELS),
        "pass_definition": PASS_DEFINITION,
        "notes": [
            "eval_commit is the skill revision attached for the skill-on attempts.",
            "When the grader tree is not a git checkout, the SHA is supplied at publication.",
            "Skill-absent attempts used PyMC; skill-attached attempts used Stan via CmdStanPy. The two are not separable in this design.",
            "The blank-response observation guidance was written in this same revision.",
            "Three of twelve paired attempts differed, all in the same direction. This is one run, not an effect-size estimate.",
        ],
    }
    payload.update(suite_from_cells(cells, TASK_IDS))
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
        Document from ``build_public_on_off`` or ``pending_public_on_off``.
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
    status = str(payload.get("status") or "")
    if status not in {"not_yet_run", "complete"}:
        problems.append("status must be not_yet_run or complete")
        return problems
    required = REQUIRED_PENDING_KEYS if status == "not_yet_run" else REQUIRED_COMPLETE_KEYS
    for key in required:
        if key not in payload:
            problems.append(f"missing top-level key {key}")
    if problems:
        return problems
    n_attempts = int(payload["attempts_per_cell"])
    if n_attempts != ATTEMPTS_PER_CELL:
        problems.append(f"attempts_per_cell must be {ATTEMPTS_PER_CELL}")
    definition = str(payload.get("pass_definition") or "")
    if "Band A" in definition or "Band B" in definition:
        problems.append("pass_definition must not use Band A / Band B")
    task_ids = list(payload.get("task_ids") or [])
    if task_ids != list(TASK_IDS):
        problems.append(f"task_ids {task_ids} != {list(TASK_IDS)}")
    for task_id in task_ids:
        if not (packs_root / task_id / "prompt.md").is_file():
            problems.append(f"missing tracked task {task_id}")
        if not (packs_root / task_id / "data.csv").is_file():
            problems.append(f"missing data.csv for {task_id}")
    if "model" in payload or "solver" in payload:
        problems.append("public file must not contain a solver field")
    if status == "not_yet_run":
        return problems
    if not _eval_commit_ok(str(payload.get("eval_commit") or "")):
        problems.append("eval_commit must be a git SHA, not unknown")
    for key in REQUIRED_SUITE_KEYS:
        if key not in payload:
            problems.append(f"missing {key}")
    attempts = payload.get("attempt_successes") or {}
    out_of = int(attempts.get("out_of") or 0)
    if out_of != n_attempts * len(task_ids):
        problems.append(f"attempt out_of {out_of} is inconsistent")
    for cond in ("off", "on"):
        successes = int(attempts.get(cond) or 0)
        if not 0 <= successes <= out_of:
            problems.append(f"{cond} successes out of range")
        expected_rate = _rate(successes / out_of) if out_of else 0.0
        reported = _rate(float((payload.get("attempt_pass_at_1") or {}).get(cond) or 0.0))
        if expected_rate != reported:
            problems.append(f"{cond} pass^1 {reported} != {expected_rate}")
    rows = list(payload.get("tasks") or [])
    row_ids = [str(row.get("id")) for row in rows]
    if row_ids != task_ids:
        problems.append(f"task rows {row_ids} != {task_ids}")
    for row in rows:
        if not row.get("label"):
            problems.append(f"{row.get('id')} missing label")
        for key in (
            "off_attempts_passing",
            "on_attempts_passing",
            "off_pass_at_1",
            "on_pass_at_1",
            "off_both_models",
            "on_both_models",
        ):
            if key not in row:
                problems.append(f"{row.get('id')} missing {key}")
    return problems


def per_cell_trees_available(cell_dir: Path = DEFAULT_CELL_DIR) -> bool:
    """True when local per-cell score files are present."""
    try:
        load_suite_cells(cell_dir, TASK_IDS)
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
        "--cell-dir",
        type=Path,
        default=DEFAULT_CELL_DIR,
        help="directory of per-cell JSON",
    )
    parser.add_argument("--out", type=Path, default=PUBLIC_PATH)
    parser.add_argument(
        "--eval-commit",
        default=None,
        help="skill revision SHA when the grader tree is not a git checkout",
    )
    parser.add_argument(
        "--pending",
        action="store_true",
        help="write the not-yet-run public document",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.check:
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        problems = validate_public_on_off(payload)
        if problems:
            raise SystemExit("\n".join(problems))
        print(str(args.out))
        return 0
    payload = pending_public_on_off() if args.pending else build_public_on_off(
        cell_dir=args.cell_dir,
        eval_commit=args.eval_commit,
    )
    problems = validate_public_on_off(payload)
    if problems:
        raise SystemExit("\n".join(problems))
    dest = write_public_on_off(payload, dest=args.out)
    print(str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
