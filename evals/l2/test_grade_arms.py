"""Four-arm grader stacks one Grok attempt and one Opus attempt."""

from __future__ import annotations

from pathlib import Path

from grade_arms import write_cells


def _fake(passed: bool) -> dict:
    return {
        "passed": passed,
        "band_a": {"passed": passed, "predicates": []},
        "band_b": {"passed": passed, "scored": True},
    }


def test_write_cells_stacks_models(tmp_path: Path) -> None:
    packs = ("E1", "H1", "A1", "K1", "J1")
    fake_arm = {p: _fake(True) for p in packs}
    arms = {
        "off-grok": fake_arm,
        "off-opus": {p: _fake(False) for p in packs},
        "on-grok": fake_arm,
        "on-opus": fake_arm,
    }
    written = write_cells(arms, tmp_path, git="deadbeef")
    e1 = __import__("json").loads((tmp_path / "E1_without.json").read_text(encoding="utf-8"))
    assert e1["successes"] == [True, False]
    assert e1["models"] == ["grok-4.6", "opus-5"]
    assert (tmp_path / "E1_with.json") in written.values()
