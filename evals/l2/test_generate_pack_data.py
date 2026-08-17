"""Fixture screens must cover every scored truth key."""

from __future__ import annotations

from generate_pack_data import check_s4, check_s6, check_s7, check_s8


def test_s8_csv_recovers_mu_and_sigma() -> None:
    assert check_s8() is True


def test_s4_csv_recovers_alpha_beta_sigma() -> None:
    assert check_s4() is True


def test_s6_csv_recovers_beta_and_ld50() -> None:
    assert check_s6() is True


def test_s7_csv_recovers_mixture_keys() -> None:
    assert check_s7() is True
