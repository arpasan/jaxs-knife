"""CLI-level grading on fixtures."""

from __future__ import annotations

from pathlib import Path

from grade import grade_trial

FIX = Path(__file__).resolve().parent / "fixtures"


def test_pass_fixture_grades_clean() -> None:
    report = grade_trial(FIX / "pass_workflow")
    assert report["band_a"]["passed"] is True
    assert report["band_b"] is None
    assert report["passed"] is True


def test_fail_fixture_grades_false() -> None:
    report = grade_trial(FIX / "fail_workflow")
    assert report["passed"] is False
    assert report["band_a"]["passed"] is False
