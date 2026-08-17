"""Development aid: keyword overlap vs SKILL.md description.

Not a pass/fail gate. Prints precision, recall, and a Stan-vs-JAX confusion
table so we can see a split pattern if one appears.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "jaxs-knife" / "SKILL.md"
LABELS = Path(__file__).resolve().parent / "trigger_eval_set.json"


def _description(skill_md: str) -> str:
    match = re.search(r"^description:\s*>\s*\n((?:  .*\n)+)", skill_md, re.M)
    if not match:
        raise ValueError("Could not parse YAML description")
    return " ".join(line.strip() for line in match.group(1).splitlines())


def _tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9.]+", text.lower()))


def _should_fire(description: str, query: str) -> bool:
    """Crude overlap: fire if the query shares a distinctive token with the description.

    Distinctive = tokens in the description that are not English stop-ish words.
    This is a development heuristic, not the real L1 (which is agent-side).
    """
    stop = {
        "a", "an", "the", "and", "or", "for", "of", "to", "in", "on", "with",
        "when", "use", "do", "not", "that", "this", "from", "by", "is", "are",
        "be", "as", "if", "then", "than", "into", "over", "under",
    }
    desc = _tokens(description) - stop
    query_toks = _tokens(query)
    distinctive = {
        "stan", "cmdstan", "cmdstanpy", "nutpie", "jax", "blackjax",
        "logdensity", "jacobian", "divergences", "r-hat", "rhat", "ess",
        "psis", "loo", "hierarchical", "funnel", "report.md", "generated",
        "quantities", "bridgestan", ".stan",
    }
    # Fire if the query mentions a skill-specific term that the description also owns,
    # or a generic Bayesian workflow term that the description lists.
    workflow = {
        "hierarchical", "divergences", "r-hat", "rhat", "ess", "loo",
        "prior", "predictive", "report.md", "elpd", "stacking", "pooling",
    }
    overlap = (query_toks & desc & (distinctive | workflow))
    # Also fire on file/engine names even if hyphenation differs
    text = query.lower()
    named = any(
        key in text
        for key in (
            ".stan", "cmdstan", "cmdstanpy", "blackjax", "logdensity",
            "nutpie[stan]", "generated quantities", "report.md",
        )
    )
    return bool(overlap) or named


def score() -> Dict[str, object]:
    description = _description(SKILL.read_text(encoding="utf-8"))
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    tp = fp = fn = tn = 0
    by_engine: Dict[str, List[Tuple[str, bool]]] = {"stan": [], "jax": [], "unspecified": []}
    misses: List[str] = []
    false_alarms: List[str] = []

    for row in labels["positives"]:
        fired = _should_fire(description, row["text"])
        by_engine.setdefault(row["engine"], []).append((row["id"], fired))
        if fired:
            tp += 1
        else:
            fn += 1
            misses.append(row["id"])

    for row in labels["negatives"]:
        fired = _should_fire(description, row["text"])
        if fired:
            fp += 1
            false_alarms.append(row["id"])
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    engine_hits = {
        engine: {"hit": sum(1 for _, f in rows if f), "n": len(rows)}
        for engine, rows in by_engine.items()
    }
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "misses": misses,
        "false_alarms": false_alarms,
        "engine_hits": engine_hits,
        "split_signal": _split_signal(engine_hits, false_alarms),
    }


def _split_signal(engine_hits: Dict[str, Dict[str, int]], false_alarms: List[str]) -> str:
    stan = engine_hits.get("stan", {"hit": 0, "n": 0})
    jax = engine_hits.get("jax", {"hit": 0, "n": 0})
    if stan["n"] and jax["n"] and stan["hit"] == stan["n"] and jax["hit"] == 0:
        return "JAX-only prompts systematically miss"
    if stan["n"] and jax["n"] and jax["hit"] == jax["n"] and stan["hit"] == 0:
        return "Stan-only prompts systematically miss"
    if false_alarms:
        return "False positives on the negative set — tighten the description"
    return "No engine-specific miss pattern"


def main() -> None:
    print(json.dumps(score(), indent=2))


if __name__ == "__main__":
    main()
