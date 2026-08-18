"""Interpret diagnose / calibration JSON into qualitative ratings.

Library functions do not print. CLI ``main`` writes JSON and a short summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==================================================
# House thresholds (not exposed in report prose)
# ==================================================

RHAT_OK = 1.01
DIVERGENCE_FAIR = 0.005
PARETO_K_OK = 0.5
PARETO_K_FAIR = 0.7
COVERAGE_DEVIATION_FAIR = 0.05
PSENSE_OK = 0.05
PSENSE_FAIR = 0.10


# ==================================================
# Ratings
# ==================================================


def _rate_convergence(conv: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Return (rating, problematic_param_names) for convergence."""
    if not conv or conv.get("all_ok"):
        return "excellent", []

    issues: List[str] = []
    rhat = conv.get("rhat", {})
    ess_b = conv.get("ess_bulk", {})
    ess_t = conv.get("ess_tail", {})
    div = conv.get("divergences", {})
    bfmi = conv.get("bfmi", {})
    td = conv.get("treedepth", {})

    if not rhat.get("ok", True):
        issues.extend(rhat.get("problematic_params", []))
    if not ess_b.get("ok", True):
        issues.extend(ess_b.get("problematic_params", []))
    if not ess_t.get("ok", True):
        issues.extend(ess_t.get("problematic_params", []))
    if not div.get("ok", True):
        if div.get("recorded") is False:
            issues.append("divergences=unknown")
        else:
            issues.append(f"divergences={div.get('count', 0)}")
    if not bfmi.get("ok", True):
        issues.append("low E-BFMI")
    if not td.get("ok", True):
        issues.append("max treedepth saturation")

    issues = list(dict.fromkeys(issues))
    n_div = div.get("count") if div else 0
    n_div = 0 if n_div is None else n_div
    div_pct = div.get("pct") if div else 0.0
    div_pct = 0.0 if div_pct is None else div_pct
    rhat_max = rhat.get("max") if rhat else None

    if div.get("recorded") is False:
        rating = "poor"
    elif n_div > 0 and div_pct > DIVERGENCE_FAIR * 100:
        rating = "poor"
    elif rhat_max is not None and rhat_max > 1.05:
        rating = "poor"
    elif issues:
        rating = "fair"
    else:
        rating = "good"
    return rating, issues


def _rate_loo(loo: Dict[str, Any]) -> str:
    """Qualitative rating for PSIS-LOO Pareto-k."""
    if not loo or not loo.get("computed"):
        return "not computed"
    pk = loo.get("pareto_k", {})
    n_bad = pk.get("n_bad", 0)
    n_nonfinite = pk.get("n_nonfinite", 0)
    pk_max = pk.get("max")
    if n_bad > 0 or n_nonfinite > 0:
        return "poor"
    if pk_max is None:
        return "not computed"
    if pk_max <= PARETO_K_OK:
        return "excellent"
    if pk_max <= PARETO_K_FAIR:
        return "fair"
    return "poor"


def _rate_calibration(cal: Dict[str, Any]) -> Tuple[str, str]:
    """Return (rating, diagnosis) for calibration."""
    if not cal:
        return "not computed", ""
    assessment = cal.get("assessment", {})
    diagnosis = assessment.get("calibration_diagnosis", "")
    well_cal = assessment.get("well_calibrated", False)
    mean_dev = abs(assessment.get("mean_coverage_deviation", 0.0))
    if well_cal:
        return "excellent", diagnosis
    if mean_dev <= COVERAGE_DEVIATION_FAIR:
        return "fair", diagnosis
    return "poor", diagnosis


def _rate_psense(ps: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Return (rating, flagged_params) for prior sensitivity."""
    if not ps:
        return "not computed", []
    flagged: List[str] = []
    max_prior_cjs = 0.0
    for param, vals in ps.items():
        if isinstance(vals, dict):
            prior = float(vals.get("prior", 0.0))
            max_prior_cjs = max(max_prior_cjs, prior)
            if prior > PSENSE_FAIR:
                flagged.append(param)
    if max_prior_cjs <= PSENSE_OK:
        return "low sensitivity", flagged
    if max_prior_cjs <= PSENSE_FAIR:
        return "moderate sensitivity", flagged
    return "strong sensitivity", flagged


def _build_summary(report: Dict[str, Any]) -> Dict[str, str]:
    """One short sentence per section for report Assessment lines."""
    summary: Dict[str, str] = {}
    if "convergence" in report:
        rating = report["convergence"]["rating"]
        if rating == "excellent":
            summary["convergence"] = (
                "All convergence diagnostics passed "
                "(R-hat ≤ 1.01, ESS adequate, no divergences)."
            )
        elif rating == "good":
            summary["convergence"] = (
                "Convergence diagnostics broadly pass; minor flags worth noting."
            )
        elif rating == "fair":
            params = ", ".join(report["convergence"]["problematic_params"][:3]) or (
                "some parameters"
            )
            summary["convergence"] = (
                f"Convergence is fair — flags on {params}. "
                "Do not trust the posterior until inspected."
            )
        else:
            params = ", ".join(report["convergence"]["problematic_params"][:3]) or (
                "multiple parameters"
            )
            summary["convergence"] = (
                f"Poor convergence on {params}. "
                "Do not interpret the posterior until resolved."
            )
    if "loo" in report:
        summary["loo"] = f"LOO Pareto-k: {report['loo']['rating']}."
    if "calibration" in report:
        rating = report["calibration"]["rating"]
        diag = report["calibration"].get("diagnosis", "")
        summary["calibration"] = (
            f"Calibration is {rating}" + (f" — {diag}." if diag else ".")
        )
    if "psense" in report:
        flagged = (
            ", ".join(report["psense"]["flagged_params"])
            if report["psense"]["flagged_params"]
            else "none"
        )
        summary["psense"] = (
            f"Prior sensitivity: {report['psense']['rating']} (flagged: {flagged})."
        )
    return summary


def check_diagnostics(
    diagnostics: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    psense: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Interpret diagnostic JSON into qualitative assessments.

    Parameters
    ----------
    diagnostics : dict, optional
        Output of ``diagnose_model.generate_report``.
    calibration : dict, optional
        Output of ``calibration_check``.
    psense : dict, optional
        ``{param: {"prior": cjs, "likelihood": cjs}}``.

    Returns
    -------
    Dict[str, Any]
        Sections plus ``summary``.
    """
    report: Dict[str, Any] = {}
    if diagnostics is not None:
        conv_rating, conv_issues = _rate_convergence(diagnostics.get("convergence", {}))
        report["convergence"] = {
            "rating": conv_rating,
            "problematic_params": conv_issues,
        }
        loo_in = diagnostics.get("loo", {})
        report["loo"] = {
            "rating": _rate_loo(loo_in),
            "pareto_k": loo_in.get("pareto_k", {}),
        }
        ppc = diagnostics.get("posterior_predictive", {})
        report["posterior_predictive"] = {"available": bool(ppc.get("available", False))}
    if calibration is not None:
        cal_rating, cal_diagnosis = _rate_calibration(calibration)
        report["calibration"] = {"rating": cal_rating, "diagnosis": cal_diagnosis}
    if psense is not None:
        ps_rating, ps_flagged = _rate_psense(psense)
        report["psense"] = {"rating": ps_rating, "flagged_params": ps_flagged}
    report["summary"] = _build_summary(report)
    return report


def suggest_next_steps(report: Dict[str, Any]) -> List[str]:
    """Ordered next steps. Most-critical issues first.

    Parameters
    ----------
    report : dict
        Output of ``check_diagnostics``.

    Returns
    -------
    List[str]
        Actionable steps for ``report.md``.
    """
    steps: List[str] = []
    conv = report.get("convergence", {})
    if conv.get("rating") in ("poor", "fair"):
        params = conv.get("problematic_params", [])
        has_divergences = any("divergence" in str(p).lower() for p in params)
        non_param = ("divergence", "e-bfmi", "treedepth", "=")
        named_params = [
            p for p in params if not any(tok in str(p).lower() for tok in non_param)
        ]
        if any("unknown" in str(p).lower() for p in params):
            steps.append(
                "sample_stats.diverging is missing — store BlackJAX is_divergent "
                "(or CmdStan divergences) on InferenceData. A missing flag is not "
                "zero divergences. Do not interpret."
            )
        elif has_divergences:
            steps.append(
                "Divergences detected — reparameterize first (non-centered hierarchical "
                "scales; constrained decls or explicit Jacobian; Gamma/Exponential "
                "instead of HalfCauchy on scales). Then raise adapt_delta / target_accept "
                "to 0.95–0.99. Do not interpret until divergences are gone."
            )
        if named_params:
            preview = ", ".join(named_params[:3])
            steps.append(
                f"R-hat or ESS flags on {preview} — more draws, Pathfinder as NUTS "
                "init (label Pathfinder as approximation), or check multimodality "
                "with az.plot_rank / pair plots."
            )

    cal = report.get("calibration", {})
    if cal.get("rating") == "poor":
        diag = cal.get("diagnosis", "")
        if "over-confident" in diag:
            steps.append(
                "Calibration is over-confident — likelihood too narrow. "
                "Student-t for heavy tails, neg_binomial for overdispersed counts, "
                "or hierarchical variance if groups differ."
            )
        elif "under-confident" in diag:
            steps.append(
                "Calibration is under-confident — tighten priors that dominate, "
                "or simplify an overparameterized model."
            )
        else:
            steps.append(
                "Calibration failed — re-examine the likelihood and prior predictive "
                "range before interpreting posteriors."
            )
    elif cal.get("rating") == "fair":
        steps.append(
            "Calibration is fair — consider a heavier-tailed likelihood or a "
            "sensitivity check on influential observations."
        )

    loo = report.get("loo", {})
    loo_rating = loo.get("rating", "not computed")
    loo_pk = loo.get("pareto_k", {})
    n_high = loo_pk.get("n_bad", 0)
    n_nonfinite = loo_pk.get("n_nonfinite", 0)
    if loo_rating == "poor":
        if n_high:
            steps.append(
                "LOO Pareto-k > 0.7 — inspect those points; consider Student-t "
                "or K-fold. Moment matching needs a callable log density, not a "
                "CmdStan CSV. Use az.loo(..., pointwise=True)."
            )
        if n_nonfinite:
            steps.append(
                f"LOO could not be estimated for {n_nonfinite} observation(s) "
                "(non-finite Pareto-k). Fix convergence first."
            )
        if not n_high and not n_nonfinite:
            steps.append("LOO flags a problem — inspect pointwise Pareto-k.")
    elif loo_rating == "fair":
        steps.append(
            "LOO Pareto-k between 0.5 and 0.7 — identify the points; "
            "consider a robust likelihood if they are tail-heavy."
        )

    if not report.get("posterior_predictive", {}).get("available", True):
        steps.append(
            "No posterior_predictive group — add generated quantities (Stan) or "
            "vmap the observation model (JAX). Do not rewrite the likelihood in numpy."
        )

    ps = report.get("psense", {})
    if ps.get("rating") == "strong sensitivity":
        flagged = ", ".join(ps.get("flagged_params", []))
        steps.append(
            f"Strong prior sensitivity on {flagged} — justify the prior or widen "
            "it and refit. Report both versions if the conclusion changes."
        )
    elif ps.get("rating") == "moderate sensitivity":
        flagged = ", ".join(ps.get("flagged_params", []))
        steps.append(
            f"Moderate prior sensitivity on {flagged} — note in the report."
        )

    if not steps:
        steps.append(
            "All diagnostics are within acceptable bounds — proceed to "
            "interpretation. Use probability language; document assumptions."
        )
        if report.get("psense", {}).get("rating") == "not computed":
            steps.append(
                "Consider power-scaling sensitivity (psense_summary) before "
                "publication if conclusions are decision-relevant."
            )
    return steps


def _load_optional(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpret Bayesian diagnostics and suggest next steps"
    )
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--psense", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    diagnostics = _load_optional(args.diagnostics)
    if diagnostics is None:
        sys.stdout.write(
            json.dumps({"error": f"Could not load diagnostics: {args.diagnostics}"})
        )
        sys.exit(1)

    report = check_diagnostics(
        diagnostics=diagnostics,
        calibration=_load_optional(args.calibration),
        psense=_load_optional(args.psense),
    )
    report["next_steps"] = suggest_next_steps(report)
    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
