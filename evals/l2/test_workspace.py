"""Isolated workspaces copy prompt and data only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from isolation import IsolationError
from workspace import pack_truth, prepare_workspace


def test_without_skill_copies_prompt_and_data_only(tmp_path: Path) -> None:
    dest = tmp_path / "S1-without"
    prepare_workspace("S1", dest, condition="without")
    names = {p.name for p in dest.iterdir()}
    assert names == {"prompt.md", "data.csv"}
    assert not (dest / "meta.json").exists()
    assert not (dest / ".cursor").exists()
    assert not (dest / "evals").exists()


def test_with_skill_does_not_copy_evals() -> None:
    # Stay inside the repo: some sandboxes block creating `.cursor` under /tmp.
    dest = (
        Path(__file__).resolve().parent / "local_runs" / "_pytest" / "S1-with"
    )
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        prepare_workspace("S1", dest, condition="with")
        skill = dest / ".cursor" / "skills" / "stan-jax-workflow"
        assert (skill / "SKILL.md").is_file()
        assert not (dest / "evals").exists()
        assert not (skill / "evals").exists()
        assert not list(dest.rglob("rubric.json"))
        assert not list(dest.rglob("meta.json"))
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def test_refuses_nonempty_destination(tmp_path: Path) -> None:
    dest = tmp_path / "used"
    dest.mkdir()
    (dest / "already.txt").write_text("x", encoding="utf-8")
    with pytest.raises(IsolationError):
        prepare_workspace("S1", dest, condition="without")


def test_truth_stays_in_pack_meta() -> None:
    truth = pack_truth("S1")
    assert truth is not None
    assert truth["beta"] == 0.9
    assert pack_truth("S2") is None


def test_run_trial_prepare_and_grade(tmp_path: Path) -> None:
    from run_trial import main

    rc = main(
        [
            "--pack",
            "S1",
            "--condition",
            "without",
            "--n",
            "1",
            "--run-root",
            str(tmp_path),
            "--grade",
        ]
    )
    assert rc == 0
    batch = tmp_path / "S1" / "without" / "batch.json"
    payload = json.loads(batch.read_text(encoding="utf-8"))
    assert "receipts" in payload
    assert payload["grade"]["successes"] == [False]
