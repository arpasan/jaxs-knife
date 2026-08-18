"""Discovery prompts must not name the skill or leak evaluator files."""

from __future__ import annotations

from pathlib import Path

PACKS = Path(__file__).resolve().parent / "packs"
FORBIDDEN = (
    "jaxs-knife",
    "rubric.json",
    "eval_metadata.json",
    "truth.json",
    "meta.json",
)


def test_pack_prompts_do_not_name_the_skill() -> None:
    prompts = sorted(PACKS.glob("*/prompt.md"))
    assert prompts, "no pack prompts found"
    for path in prompts:
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN:
            assert token not in text, f"{path}: contains {token}"


def test_expected_packs_exist() -> None:
    ids = {p.name for p in PACKS.iterdir() if p.is_dir()}
    assert {
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "M1",
        "F1",
        "X1",
        "C1",
    } <= ids
