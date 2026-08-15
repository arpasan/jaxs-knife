"""pass^k algebra."""

from __future__ import annotations

from passk import pass_at_k, skill_delta


def test_all_pass() -> None:
    assert pass_at_k([True, True, True], 3) == 1.0


def test_two_of_three() -> None:
    # C(2,2)/C(3,2) = 1/3 for k=2
    assert abs(pass_at_k([True, True, False], 2) - (1.0 / 3.0)) < 1e-12


def test_k_exceeds_n() -> None:
    assert pass_at_k([True, True], 3) == 0.0


def test_skill_delta_positive() -> None:
    delta = skill_delta([True, True, True], [True, False, False], k=1)
    assert delta > 0
