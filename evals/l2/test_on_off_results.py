"""Public on/off scores have a schema and match tracked tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarize_on_off import (
    DEFAULT_CELL_DIR,
    PUBLIC_PATH,
    TASK_IDS,
    pending_public_on_off,
    per_cell_trees_available,
    build_public_on_off,
    validate_public_on_off,
)


def test_committed_on_off_schema() -> None:
    payload = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    assert validate_public_on_off(payload) == []
    assert "model" not in payload
    assert "Band A" not in payload["pass_definition"]
    assert payload["task_ids"] == list(TASK_IDS)
    assert payload["attempts_per_cell"] == 2
    assert payload["status"] in {"not_yet_run", "complete"}


def test_pending_document_validates() -> None:
    payload = pending_public_on_off()
    assert validate_public_on_off(payload) == []
    assert payload["status"] == "not_yet_run"


def test_rebuild_matches_committed_when_local_cells_exist() -> None:
    if not per_cell_trees_available(DEFAULT_CELL_DIR):
        pytest.skip("local per-cell score files are not present")
    rebuilt = build_public_on_off(cell_dir=DEFAULT_CELL_DIR)
    committed = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    if committed.get("status") != "complete":
        pytest.skip("committed public file is still not_yet_run")
    for key in (
        "status",
        "date",
        "eval_commit",
        "attempts_per_cell",
        "pass_definition",
        "task_ids",
        "attempt_successes",
    ):
        assert rebuilt[key] == committed[key]


def test_sealed_tasks_are_on_disk() -> None:
    packs = Path(__file__).resolve().parent / "packs"
    for pack_id in TASK_IDS:
        assert (packs / pack_id / "prompt.md").is_file()
        assert (packs / pack_id / "data.csv").is_file()
        assert (packs / pack_id / "meta.json").is_file()
