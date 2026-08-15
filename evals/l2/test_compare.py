"""Portable pass^k comparison of two graded batches."""

from __future__ import annotations

import json
from pathlib import Path

from compare import compare_batches


def _batch(path: Path, successes: list[bool]) -> Path:
    path.write_text(
        json.dumps({"grade": {"successes": successes}}),
        encoding="utf-8",
    )
    return path


def test_compare_delta(tmp_path: Path) -> None:
    ours = _batch(tmp_path / "ours.json", [True, True, True])
    other = _batch(tmp_path / "other.json", [True, False, False])
    report = compare_batches(ours, other, k=1)
    assert report["ours"]["pass_at_k"] == 1.0
    assert report["delta_ours_minus_other"] > 0
