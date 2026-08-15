"""Deterministic Band A predicates on a trial directory (not the skill tree)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_PVALUE = re.compile(
    r"\b(p\s*[<>=]\s*0\.\d+|p-value|p value|statistically significant|significant effect|reject the null)\b",
    re.I,
)
_PRIOR_PRED = re.compile(r"prior\s+predictive", re.I)
_SAMPLE = re.compile(r"\b(sample\(|\.sample\(|pm\.sample|nuts|mcmc)\b", re.I)
_LIMIT = re.compile(r"\b(limitations?|threats?)\b", re.I)
_HDI = re.compile(r"\bhdi\b", re.I)
_50 = re.compile(r"\b50\s*%", re.I)
_94 = re.compile(r"\b94\s*%", re.I)
_RHAT = re.compile(r"1\.01")
_DIVERGE = re.compile(r"divergenc", re.I)
_REFUSE = re.compile(
    r"(do not interpret|must not be interpreted|refuse to interpret|not interpret the posterior)",
    re.I,
)
_NETCDF = re.compile(r"(inference_data\.nc|to_netcdf|write_netcdf)", re.I)
_GQ = re.compile(r"(generated quantities|y_rep|log_lik|jax\.vmap|vmap\()", re.I)
_NUMPY_LIK = re.compile(r"(numpy.*normal\(|np\.random\.normal\(.*mu|rewrite.{0,20}likelihood)", re.I)
_CONSTRAINT = re.compile(r"(<lower\s*=|jacobian\s*\+=|jnp\.exp\(|log_abs_det)", re.I)
_PATHFINDER = re.compile(r"pathfinder", re.I)
_APPROX = re.compile(r"(approximation|warm[- ]?start|init(?:ialization)?)", re.I)


def _iter_text_files(root: Path) -> Iterable[Tuple[Path, str]]:
    skip = {".nc", ".png", ".npy", ".pdf", ".so", ".o"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in skip:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        yield path, text


def _corpus(root: Path) -> str:
    return "\n".join(text for _, text in _iter_text_files(root))


def _code_order_prior_before_sample(root: Path) -> bool:
    for path, text in _iter_text_files(root):
        if path.suffix not in {".py", ".stan", ".md"}:
            continue
        prior = _PRIOR_PRED.search(text)
        sample = _SAMPLE.search(text)
        if prior and sample and prior.start() < sample.start():
            return True
        if path.suffix == ".md" and prior:
            return True
    return False


def _divergences_count(root: Path) -> int | None:
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        conv = payload.get("convergence") if isinstance(payload, dict) else None
        if not isinstance(conv, dict):
            continue
        div = conv.get("divergences")
        if isinstance(div, dict) and "count" in div:
            return int(div["count"])
    return None


def evaluate_band_a(trial_dir: Path) -> Dict[str, Any]:
    """Score Band A predicates.

    Parameters
    ----------
    trial_dir : Path
        Agent output directory. Must not contain this rubric.

    Returns
    -------
    Dict[str, Any]
        ``predicates``, ``passed``, ``n_pass``, ``n_total``.
    """
    root = trial_dir.resolve()
    corpus = _corpus(root)
    reports = list(root.rglob("report.md"))
    n_div = _divergences_count(root)

    predicates: List[Dict[str, Any]] = []

    def add(pid: str, ok: bool, evidence: str) -> None:
        predicates.append({"id": pid, "ok": bool(ok), "evidence": evidence})

    add("report_exists", bool(reports), reports[0].name if reports else "missing report.md")
    add("limitations", bool(_LIMIT.search(corpus)), "limitations/threats heading" if _LIMIT.search(corpus) else "absent")
    add(
        "probability_language",
        _PVALUE.search(corpus) is None,
        "clean" if _PVALUE.search(corpus) is None else _PVALUE.search(corpus).group(0),
    )
    add(
        "intervals_50_94",
        bool(_50.search(corpus) and _94.search(corpus) and _HDI.search(corpus)),
        "50% and 94% HDI" if (_50.search(corpus) and _94.search(corpus)) else "missing 50/94 HDI",
    )
    add(
        "prior_predictive",
        _code_order_prior_before_sample(root),
        "prior predictive before sampling"
        if _code_order_prior_before_sample(root)
        else "prior predictive missing or after sampling",
    )
    add("rhat_1_01", bool(_RHAT.search(corpus)), "1.01" if _RHAT.search(corpus) else "R-hat 1.01 not stated")
    if n_div is None:
        add(
            "refuse_divergences",
            bool(_DIVERGE.search(corpus) and _REFUSE.search(corpus)) or bool(re.search(r"\b0 diverg", corpus, re.I)),
            "divergence policy stated" if _DIVERGE.search(corpus) else "divergences not discussed",
        )
    elif n_div > 0:
        add("refuse_divergences", bool(_REFUSE.search(corpus)), f"divergences={n_div}")
    else:
        add("refuse_divergences", True, "diagnostics report 0 divergences")
    add(
        "draws_saved",
        bool(_NETCDF.search(corpus)) or bool(list(root.rglob("*.nc"))),
        "netcdf present" if list(root.rglob("*.nc")) else "save not evidenced",
    )
    numpy_rewrite = bool(_NUMPY_LIK.search(corpus)) and not bool(_GQ.search(corpus))
    add("gq_or_vmap", bool(_GQ.search(corpus)) and not numpy_rewrite, "GQ/vmap" if _GQ.search(corpus) else "no GQ/vmap")
    add("constraint_ok", bool(_CONSTRAINT.search(corpus)), "constraint/jacobian" if _CONSTRAINT.search(corpus) else "absent")
    if _PATHFINDER.search(corpus):
        add("pathfinder_labeled", bool(_APPROX.search(corpus)), "pathfinder mentioned")
    else:
        add("pathfinder_labeled", True, "pathfinder not used")

    n_pass = sum(1 for p in predicates if p["ok"])
    return {
        "predicates": predicates,
        "n_pass": n_pass,
        "n_total": len(predicates),
        "passed": n_pass == len(predicates),
    }
