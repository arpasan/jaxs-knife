"""Results JSON is produced by emit_results, not by hand."""

from __future__ import annotations

from emit_results import emit_cell


def test_emit_cell_passk_and_failed_ids() -> None:
    grade = {
        "successes": [True, False, False],
        "trials": [
            {
                "passed": True,
                "band_a": {
                    "passed": True,
                    "predicates": [{"id": "rhat_1_01", "ok": True}],
                },
                "band_b": {"passed": True, "scored": True},
            },
            {
                "passed": False,
                "band_a": {
                    "passed": False,
                    "predicates": [{"id": "rhat_1_01", "ok": False}],
                },
                "band_b": {"passed": True, "scored": True},
            },
            {
                "passed": False,
                "band_a": {
                    "passed": False,
                    "predicates": [{"id": "limitations", "ok": False}],
                },
                "band_b": {"passed": False, "scored": True},
            },
        ],
    }
    payload = emit_cell(
        pack="E1",
        condition="without",
        model="test-model",
        git="deadbeef",
        grade=grade,
    )
    assert payload["pass_at_1"] == 1.0 / 3.0
    assert payload["pass_at_3"] == 0.0
    assert payload["trials"][1]["band_a_failed"] == ["rhat_1_01"]
    assert payload["band_a_successes"] == [True, False, False]
