"""τ-bench style pass^k: P(all k independent trials succeed)."""

from __future__ import annotations

from math import comb
from typing import Sequence


def pass_at_k(successes: Sequence[bool], k: int) -> float:
    """Unbiased pass^k from ``n`` Bernoulli trials.

    Parameters
    ----------
    successes : Sequence[bool]
        One entry per trial.
    k : int
        Number of successes required in a random k-subset.

    Returns
    -------
    float
        ``C(c, k) / C(n, k)`` when ``n >= k``, else 0.0.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    n = len(successes)
    if n < k:
        return 0.0
    c = sum(1 for s in successes if s)
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def skill_delta(with_skill: Sequence[bool], without_skill: Sequence[bool], k: int) -> float:
    """pass^k(with) − pass^k(without).

    Parameters
    ----------
    with_skill, without_skill : Sequence[bool]
        Trial outcomes.
    k : int
        pass^k order.

    Returns
    -------
    float
        Skill lift on the pass^k scale.
    """
    return pass_at_k(with_skill, k) - pass_at_k(without_skill, k)
