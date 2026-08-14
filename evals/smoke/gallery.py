"""Local-only plots for inspecting a live smoke run. Not a pass/fail gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def _style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["hatch.linewidth"] = 0.5
    plt.rc("font", family="monospace")
    sns.set_style("white")


def _palette() -> list:
    import seaborn as sns

    return list(sns.color_palette("mako_r", 10)[::3])


def _hdi(draws: np.ndarray, prob: float) -> tuple[float, float]:
    x = np.sort(np.asarray(draws, dtype=float).ravel())
    n = x.size
    n_in = max(int(np.ceil(prob * n)), 1)
    width = x[n_in - 1 :] - x[: n - n_in + 1]
    i = int(np.argmin(width))
    return float(x[i]), float(x[i + n_in - 1])


def _posterior_draws(idata: Any, name: str) -> np.ndarray:
    return np.asarray(idata.posterior[name].values)


def write_engine_plots(idata: Any, out_dir: Path) -> None:
    """Write trace, forest, and PPC figures for one engine.

    Parameters
    ----------
    idata : Any
        InferenceData / DataTree.
    out_dir : Path
        Engine artifact directory.
    """
    import matplotlib.pyplot as plt

    _style()
    colors = _palette()
    out_dir.mkdir(parents=True, exist_ok=True)

    mu = _posterior_draws(idata, "mu")
    sigma = _posterior_draws(idata, "sigma")
    n_chains, n_draws = mu.shape

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    for c in range(n_chains):
        axes[0].plot(mu[c], color=colors[c % len(colors)], lw=0.6, alpha=0.85)
        axes[1].plot(sigma[c], color=colors[c % len(colors)], lw=0.6, alpha=0.85)
    axes[0].set_ylabel("mu", fontsize=12)
    axes[1].set_ylabel("sigma", fontsize=12)
    axes[1].set_xlabel("draw", fontsize=12)
    axes[0].set_title("trace", fontsize=13)
    for ax in axes:
        ax.tick_params(labelsize=10)
        sns_despine(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "trace.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    rows = [("mu", mu.ravel()), ("sigma", sigma.ravel())]
    for i, (label, draws) in enumerate(rows):
        lo80, hi80 = _hdi(draws, 0.80)
        lo94, hi94 = _hdi(draws, 0.94)
        y = len(rows) - 1 - i
        ax.plot([lo94, hi94], [y, y], color=colors[0], lw=1.2, solid_capstyle="butt")
        ax.plot([lo80, hi80], [y, y], color=colors[0], lw=4.0, solid_capstyle="butt")
        ax.plot(np.mean(draws), y, "o", mfc="none", mec=colors[0], ms=6, mew=0.8)
        ax.text(-0.15, y, label, ha="right", va="center", fontsize=11, transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_xlabel("value (bar = 80% HDI, whisker = 94% HDI)", fontsize=11)
    ax.set_title("posterior", fontsize=13)
    ax.tick_params(labelsize=10)
    sns_despine(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "forest.png", bbox_inches="tight")
    plt.close(fig)

    obs_group = idata.observed_data
    obs_name = "y" if "y" in obs_group.data_vars else list(obs_group.data_vars)[0]
    y_obs = np.asarray(obs_group[obs_name].values).ravel()
    pp_group = idata.posterior_predictive
    pp_name = (
        "y_rep"
        if "y_rep" in pp_group.data_vars
        else ("y" if "y" in pp_group.data_vars else list(pp_group.data_vars)[0])
    )
    y_rep = np.asarray(pp_group[pp_name].values).ravel()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(
        y_rep,
        bins=36,
        density=True,
        histtype="step",
        hatch="////",
        color=colors[0],
        linewidth=0.8,
        label="y_rep",
    )
    ax.plot(
        y_obs,
        np.full_like(y_obs, 0.02),
        "o",
        mfc="none",
        mec=colors[1 % len(colors)],
        ms=5,
        mew=0.4,
        label="y",
    )
    ax.set_xlabel("y", fontsize=12)
    ax.set_ylabel("density", fontsize=12)
    ax.set_title("posterior predictive", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=10, prop={"family": "monospace"})
    sns_despine(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "ppc.png", bbox_inches="tight")
    plt.close(fig)


def sns_despine(ax: Any) -> None:
    import seaborn as sns

    sns.despine(ax=ax)


def write_gallery(
    stan_idata: Any,
    jax_idata: Any,
    stan_report: Dict[str, Any],
    jax_report: Dict[str, Any],
) -> Path:
    """Write per-engine figures and a local HTML index.

    Parameters
    ----------
    stan_idata, jax_idata : Any
        Fits.
    stan_report, jax_report : dict
        Diagnostic reports.

    Returns
    -------
    Path
        ``index.html``.
    """
    write_engine_plots(stan_idata, ARTIFACTS / "stan")
    write_engine_plots(jax_idata, ARTIFACTS / "jax")
    index = ARTIFACTS / "index.html"
    index.write_text(
        _html(stan_report, jax_report),
        encoding="utf-8",
    )
    return index


def _html(stan_report: Dict[str, Any], jax_report: Dict[str, Any]) -> str:
    def card(engine: str, report: Dict[str, Any]) -> str:
        conv = report["convergence"]["rating"]
        div = report["diagnostics"]["convergence"]["divergences"]
        return f"""
        <section>
          <h2>{engine}</h2>
          <p>convergence: <strong>{conv}</strong> · divergences: {div.get("count", "?")}</p>
          <figure><img src="{engine}/trace.png" alt="{engine} trace"><figcaption>trace</figcaption></figure>
          <figure><img src="{engine}/forest.png" alt="{engine} forest"><figcaption>80% HDI bar, 94% whisker (inspection only)</figcaption></figure>
          <figure><img src="{engine}/ppc.png" alt="{engine} ppc"><figcaption>posterior predictive</figcaption></figure>
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>live NUTS smoke</title>
  <style>
    body {{ font-family: ui-monospace, monospace; margin: 24px; color: #222; }}
    main {{ display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }}
    img {{ max-width: 100%; height: auto; }}
    figcaption {{ color: #666; font-size: 12px; }}
    h1 {{ font-size: 18px; }}
    h2 {{ font-size: 15px; }}
  </style>
</head>
<body>
  <h1>live NUTS smoke — inspection gallery (not a test gate)</h1>
  <p>Same data, two engines, real NUTS. Ratings come from the diagnostic scripts.</p>
  <main>
    {card("stan", stan_report)}
    {card("jax", jax_report)}
  </main>
</body>
</html>
"""
