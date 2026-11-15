"""How the 2022-results seed weight drives the spread of One Nation's estimated district vote.

For each --seed-weight (weight on 2022 first-preference shares in the seed; the rest is
equal party shares) the full IPF is refitted and One Nation's district shares summarised
by their variance across districts and their maximum. Weight 0 is the all-ones seed.

Run from the repo root:
    uv run python claudes_analyses/seed_weight_sensitivity.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.ipf import (  # noqa: E402
    PARTY,
    build_demogs,
    calc_party_shares,
    fit_demog_party_margins,
    fit_full,
    get_estimated_shares,
    get_population_base,
    get_seed_matrix,
    load_survey,
)
from utils.spatial import calculate_2021_to_2022_mapping, load_2021_seds, load_2022_districts  # noqa: E402
from utils.vec_results import load_first_preference_shares  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
SURVEY = "sep2026"
TARGET = "One Nation"
WEIGHTS = [0.0, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
CONVERGENCE_RATE = 1e-10

# reference palette (dataviz skill), light mode
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
SERIES, ACCENT, SURFACE = "#2a78d6", "#eb6834", "#fcfcfb"


def fit_shares(weights):
    mapping = calculate_2021_to_2022_mapping(load_2021_seds(), load_2022_districts())
    population = get_population_base("adult-citizen", mapping).sort_index()
    demogs = build_demogs(mapping, population)
    districts = population.index

    demographics, topline = load_survey(SURVEY)
    pct = calc_party_shares(demographics, parties=PARTY)
    topline_p = population.sum() * topline.to_numpy(dtype=float) / topline.sum()
    ipf_mats = fit_demog_party_margins(demogs, pct, topline_p, CONVERGENCE_RATE)

    shape = (len(districts), *[len(d.get_columns()) for d in demogs], len(PARTY))
    shares = {}
    for w in weights:
        method = "all-ones" if w == 0 else "2022-results"
        seed = get_seed_matrix(method, shape, list(districts), weight=w)
        X = fit_full(seed, demogs, ipf_mats, CONVERGENCE_RATE)
        shares[w] = get_estimated_shares(X, demogs, districts)[TARGET] * 100
        print(f"weight={w:<5} max={shares[w].max():5.1f}%  var={shares[w].var():7.2f}")
    return pd.DataFrame(shares), topline[TARGET] / topline.sum() * 100


def plot(summary, topline_pct, actual_2022):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(summary["variance"], summary["max"], color=SERIES, lw=2, zorder=2)
    ax.scatter(summary["variance"], summary["max"], s=64, color=SERIES,
               edgecolor=SURFACE, linewidth=2, zorder=3)
    # weights below 0.3 sit on top of the all-ones seed, so they share one label
    low = summary.loc[summary.index < 0.3]
    ax.annotate(f"w = 0 to {low.index.max():g} (≈ all-ones seed)", (low["variance"].max(), low["max"].max()),
                xytext=(6, -16), textcoords="offset points", ha="left", fontsize=9, color=INK_2)
    for w, row in summary.loc[summary.index >= 0.3].iterrows():
        offset, ha = ((-4, 10), "center") if w == 0.3 else ((10, -4), "left")
        ax.annotate(f"w = {w:g}", (row["variance"], row["max"]), xytext=offset,
                    textcoords="offset points", ha=ha, fontsize=9, color=INK_2)

    ax.axhline(topline_pct, color=INK_3, lw=1, ls="--", zorder=1)
    ax.annotate(f"Sep 2026 statewide topline ({topline_pct:.1f}%)", (1, topline_pct),
                xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points",
                ha="right", fontsize=9, color=INK_3)

    ax.set_xscale("log")
    ax.set_xlabel("Variance of One Nation primary share across districts (pp², log scale)", color=INK_2)
    ax.set_ylabel("Highest One Nation primary share in any district (%)", color=INK_2)
    ax.set_title("More weight on the 2022 seed concentrates One Nation's vote in a few districts",
                 loc="left", color=INK, fontsize=12, pad=40)
    ax.text(0, 1.02, f"IPF estimates, {SURVEY} survey. w = weight on 2022 first-preference shares in the "
            f"seed (rest is equal party shares).\nIn 2022 One Nation's actual max was "
            f"{actual_2022.max():.1f}% (contested {int((actual_2022 > 0).sum())} of {len(actual_2022)} districts).",
            transform=ax.transAxes, fontsize=8.5, color=INK_3, va="bottom")

    ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.tick_params(colors=INK_2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "one_nation_variance_vs_max.png", dpi=160, facecolor=SURFACE)


def plot_distributions(shares, topline_pct):
    """Companion view: the full district distribution at each weight."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels = ["all-ones" if w == 0 else f"{w:g}" for w in shares.columns]
    ax.boxplot([shares[w] for w in shares.columns], tick_labels=labels, widths=0.5,
               medianprops=dict(color=ACCENT, lw=2), boxprops=dict(color=SERIES),
               whiskerprops=dict(color=SERIES), capprops=dict(color=SERIES),
               flierprops=dict(marker="o", markersize=4, markerfacecolor=SERIES,
                               markeredgecolor=SURFACE, alpha=0.8))
    ax.axhline(topline_pct, color=INK_3, lw=1, ls="--")
    ax.set_xlabel("Seed weight on 2022 results", color=INK_2)
    ax.set_ylabel("One Nation primary share by district (%)", color=INK_2)
    ax.set_title("One Nation district estimates by seed weight", loc="left", color=INK, fontsize=12)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "one_nation_distribution_by_weight.png", dpi=160, facecolor=SURFACE)


def main():
    shares, topline_pct = fit_shares(WEIGHTS)
    summary = pd.DataFrame({
        "variance": shares.var(),
        "std": shares.std(),
        "min": shares.min(),
        "max": shares.max(),
        "max_district": shares.idxmax(),
    }).rename_axis("seed_weight")
    summary.to_csv(OUT_DIR / "one_nation_seed_weight_summary.csv")
    shares.to_csv(OUT_DIR / "one_nation_district_shares_by_weight.csv")
    print(summary.to_string())

    actual_2022 = load_first_preference_shares()[TARGET] * 100
    plot(summary, topline_pct, actual_2022)
    plot_distributions(shares, topline_pct)


if __name__ == "__main__":
    main()
