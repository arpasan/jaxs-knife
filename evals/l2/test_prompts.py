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
    "log_mix",
    "jacobian",
    "non-centered",
    "noncentered",
    "truncated",
    "lccdf",
    "hierarchical",
    "measurement error",
    "attenuation",
    "rogan",
    "blackjax",
    "generated quantities",
    "vmap",
    "transpile",
    "write a jax",
    "jax log-density",
    "cmdstan",
    "logdensity",
    "log-density",
)
EXPECTED = {"E1", "H1", "A1", "K1", "J1"}


def test_pack_prompts_do_not_name_the_skill() -> None:
    prompts = sorted(PACKS.glob("*/prompt.md"))
    assert prompts, "no pack prompts found"
    for path in prompts:
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN:
            assert token not in text, f"{path}: contains {token}"


def test_expected_packs_exist() -> None:
    ids = {p.name for p in PACKS.iterdir() if p.is_dir()}
    assert ids == EXPECTED
