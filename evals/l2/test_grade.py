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


def test_prefers_inference_data_nc(tmp_path: Path) -> None:
    from grade import find_inference_data

    (tmp_path / "prior_predictive.nc").write_bytes(b"prior")
    (tmp_path / "inference_data_sensitivity.nc").write_bytes(b"sens")
    dest = tmp_path / "inference_data.nc"
    dest.write_bytes(b"real")
    assert find_inference_data(tmp_path) == dest


def test_fail_fixture_grades_false() -> None:
    report = grade_trial(FIX / "fail_workflow")
    assert report["passed"] is False
    assert report["band_a"]["passed"] is False
