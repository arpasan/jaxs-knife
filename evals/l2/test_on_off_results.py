"""Public on/off scores have a schema and match tracked tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarize_on_off import (
    ADDITIONAL_IDS,
    DEFAULT_CELL_DIR,
    PUBLIC_PATH,
    STANDARD_IDS,
    per_cell_trees_available,
    build_public_on_off,
    validate_public_on_off,
)


def test_committed_on_off_schema() -> None:
    payload = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    assert validate_public_on_off(payload) == []
    assert "model" not in payload
    assert "Band A" not in payload["pass_definition"]
    assert payload["eval_commit"]
    assert payload["standard"]["task_ids"] == list(STANDARD_IDS)
    assert payload["additional"]["task_ids"] == list(ADDITIONAL_IDS)
    for suite_name in ("standard", "additional"):
        for row in payload[suite_name]["tasks"]:
            assert row["label"]


def test_rebuild_matches_committed_when_local_cells_exist() -> None:
    repo = Path(__file__).resolve().parents[2]
    standard_dir = DEFAULT_CELL_DIR
    additional_dir = None
    local_root = repo / ".local"
    if local_root.is_dir():
        for path in local_root.glob("*/runs-n3"):
            if (path / "M1_with.json").is_file():
                additional_dir = path
                break
    if additional_dir is None or not per_cell_trees_available(
        standard_dir=standard_dir,
        additional_dir=additional_dir,
    ):
        pytest.skip("local per-cell score files are not present")
    rebuilt = build_public_on_off(
        standard_dir=standard_dir,
        additional_dir=additional_dir,
    )
    committed = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    for key in (
        "date",
        "eval_commit",
        "attempts_per_cell",
        "pass_definition",
        "standard",
        "additional",
    ):
        assert rebuilt[key] == committed[key]


def test_standard_and_additional_tasks_are_on_disk() -> None:
    packs = Path(__file__).resolve().parent / "packs"
    for pack_id in (*STANDARD_IDS, *ADDITIONAL_IDS):
        assert (packs / pack_id / "prompt.md").is_file()
        assert (packs / pack_id / "data.csv").is_file()
        assert (packs / pack_id / "meta.json").is_file()
