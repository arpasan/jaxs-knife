"""Evaluator artifacts must not sit in a trial directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from isolation import IsolationError, assert_sealed, contamination
from grade import grade_trial

FIX = Path(__file__).resolve().parent / "fixtures"


def test_clean_fixture_is_sealed() -> None:
    assert contamination(FIX / "pass_workflow") == []
    assert_sealed(FIX / "pass_workflow")


def test_rubric_in_trial_is_contamination(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "report.md").write_text("# ok\n", encoding="utf-8")
    (trial / "rubric.json").write_text("{}", encoding="utf-8")
    assert "rubric.json" in contamination(trial)
    with pytest.raises(IsolationError):
        assert_sealed(trial)


def test_grade_refuses_contaminated_trial(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "report.md").write_text("# ok\n", encoding="utf-8")
    (trial / "eval_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IsolationError):
        grade_trial(trial)
