"""Public on/off scores have a schema and match tracked packs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarize_on_off import (
    HOMEWORK_IDS,
    PUBLIC_PATH,
    SCIENCE_IDS,
    per_cell_trees_available,
    build_public_on_off,
    validate_public_on_off,
)


def test_committed_on_off_schema() -> None:
    payload = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    assert validate_public_on_off(payload) == []
    assert "model" not in payload
    assert payload["eval_commit"]
    assert payload["homework"]["pack_ids"] == list(HOMEWORK_IDS)
    assert payload["science"]["pack_ids"] == list(SCIENCE_IDS)


def test_rebuild_matches_committed_when_local_cells_exist() -> None:
    if not per_cell_trees_available():
        pytest.skip("local per-cell score files are not present")
    rebuilt = build_public_on_off()
    committed = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    for key in ("date", "eval_commit", "n", "headline", "homework", "science"):
        assert rebuilt[key] == committed[key]


def test_science_and_homework_packs_are_on_disk() -> None:
    packs = Path(__file__).resolve().parent / "packs"
    for pack_id in (*HOMEWORK_IDS, *SCIENCE_IDS):
        assert (packs / pack_id / "prompt.md").is_file()
        assert (packs / pack_id / "data.csv").is_file()
        assert (packs / pack_id / "meta.json").is_file()
