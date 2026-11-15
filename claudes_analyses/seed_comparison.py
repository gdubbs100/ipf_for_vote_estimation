"""Compare four seed matrices: spread and swing against 2022, and leave-one-demographic-out accuracy.

Seeds (all with --seed-weight 1 and default smoothing; the 2022 seeds with --apply-swing):
    all-ones     demographics only, no district pattern
    lower house  2022-results
    upper house  2022-upper-house
    combined     2022-combined, upper house weight 0.5

Leave one demographic out (LODO): refit keeping every census district x demographic
margin but dropping the omitted demographic's survey demographic x party margin, then
compare the demographic x party table the fit implies with the one it was meant to hit.
The seed is constant across demographics, so the only route by which the model can
recover how the omitted group votes is through where that group lives.

Run from the repo root:
    uv run python claudes_analyses/seed_comparison.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.ipf import (  # noqa: E402
    PARTY,
    build_demogs,
    calc_party_shares,
    fit_demog_party_margins,
    get_estimated_shares,
    get_population_base,
    get_seed_matrix,
    get_swing,
    load_survey,
    run_ipf,
)
from utils.spatial import calculate_2021_to_2022_mapping, load_2021_seds, load_2022_districts  # noqa: E402
from utils.vec_results import load_first_preference_shares  # noqa: E402
from utils.vec_upper_house import load_upper_house_shares  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PREFIX = "seed_comparison"
SURVEY = "sep2026"
CONVERGENCE_RATE = 1e-10

SEEDS = {
    "all-ones": dict(method="all-ones"),
    "lower house": dict(method="2022-results", swing=True),
    "upper house": dict(method="2022-upper-house", swing=True),
    "combined": dict(method="2022-combined", swing=True, upper_house_weight=0.5),
}

# party colours as in run_analysis.ipynb
PARTY_COLOURS = {"Labor": "red", "L/NP": "blue", "Greens": "green", "One Nation": "orange", "Others": "grey"}
# reference palette (dataviz skill) slots 1-4, validated light mode; used where colour = seed
SEED_COLOURS = dict(zip(SEEDS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
SEED_MARKERS = dict(zip(SEEDS, ["o", "s", "^", "D"]))
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
CATEGORY_MARKERS = ["o", "s", "^"]


# --- fitting ---

def setup():
    districts_2022 = load_2022_districts()
    mapping = calculate_2021_to_2022_mapping(load_2021_seds(), districts_2022)
    population = get_population_base("adult-citizen", mapping).sort_index()
    demogs = build_demogs(mapping, population)
    demographics, topline = load_survey(SURVEY)
    pct = calc_party_shares(demographics, parties=PARTY)
    topline_p = population.sum() * topline.to_numpy(dtype=float) / topline.sum()
    ipf_mats = fit_demog_party_margins(demogs, pct, topline_p, CONVERGENCE_RATE)
    region = districts_2022.set_index("DISTRICT")["REGION"].str.title()
    return population, demogs, topline, topline_p, ipf_mats, region


def fit(seed, demogs, ipf_mats, omit=None):
    """fit_full, optionally without the ``omit`` demographic's survey demographic x party margin."""
    party_axis = len(demogs) + 1
    aggregates, dimensions = [], []
    for ax, demog in enumerate(demogs, start=1):
        aggregates.append(demog.data.to_numpy())
        dimensions.append([0, ax])
    for ax, demog in enumerate(demogs, start=1):
        if demog.name != omit:
            aggregates.append(ipf_mats[demog.name])
            dimensions.append([ax, party_axis])
    return run_ipf(seed.astype(float, copy=True), aggregates, dimensions, CONVERGENCE_RATE)


def demog_party_shares(X, axis):
    """Party shares within each category of the demographic on ``axis``."""
    counts = X.sum(axis=tuple(a for a in range(X.ndim) if a not in (axis, X.ndim - 1)))
    return counts / counts.sum(axis=1, keepdims=True)


def run_all():
    population, demogs, topline, topline_p, ipf_mats, region = setup()
    districts = population.index
    shape = (len(districts), *[len(d.get_columns()) for d in demogs], len(PARTY))
    swing = get_swing(topline)

    shares, lodo = {}, []
    for name, spec in SEEDS.items():
        seed = get_seed_matrix(
            spec["method"], shape, list(districts), weight=1.0,
            upper_house_weight=spec.get("upper_house_weight", 0.5),
            swing=swing if spec.get("swing") else None,
        )
        X = fit(seed, demogs, ipf_mats)
        shares[name] = get_estimated_shares(X, demogs, districts)[PARTY]
        print(f"{name}: full fit done")

        for ax, demog in enumerate(demogs, start=1):
            predicted = demog_party_shares(fit(seed, demogs, ipf_mats, omit=demog.name), ax)
            target = ipf_mats[demog.name] / ipf_mats[demog.name].sum(axis=1, keepdims=True)
            for i, category in enumerate(demog.get_columns()):
                for j, party in enumerate(PARTY):
                    lodo.append({"seed": name, "demographic": demog.name, "category": category,
                                 "party": party, "target": 100 * target[i, j],
                                 "predicted": 100 * predicted[i, j]})
            print(f"{name}: LODO {demog.name} done")

    naive = pd.Series(100 * topline_p / topline_p.sum(), index=PARTY)
    lodo = pd.DataFrame(lodo)
    lodo["naive"] = lodo["party"].map(naive)
    return shares, lodo, swing, region


# --- summaries ---

def spread_table(shares, fp22):
    rows = [{"seed": "2022 actual", "party": p, "sd": 100 * fp22[p].std(), "cv": fp22[p].std() / fp22[p].mean()}
            for p in PARTY]
    for name, s in shares.items():
        for p in PARTY:
            rows.append({"seed": name, "party": p, "sd": 100 * s[p].std(), "cv": s[p].std() / s[p].mean(),
                         "rank_corr_2022": s[p].corr(fp22[p], method="spearman")})
    return pd.DataFrame(rows)


def swing_table(shares, fp22):
    rows = []
    for name, s in shares.items():
        change = 100 * (s - fp22)
        for p in PARTY:
            rows.append({"seed": name, "party": p, "mean_swing": change[p].mean(), "swing_sd": change[p].std(),
                         "min_swing": change[p].min(), "min_district": change[p].idxmin(),
                         "max_swing": change[p].max(), "max_district": change[p].idxmax()})
    return pd.DataFrame(rows)


def lodo_errors(lodo):
    err = lodo.assign(model_err=(lodo["predicted"] - lodo["target"]).abs(),
                      naive_err=(lodo["naive"] - lodo["target"]).abs())
    return err.groupby(["seed", "demographic"], sort=False)[["model_err", "naive_err"]].mean().reset_index()


# --- plots ---

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.tick_params(colors=INK_2)


def grouped_boxes(ax, series, hatched=None):
    """One group per party; boxes per series in ``series`` order. ``hatched`` names the reference series."""
    n = len(series)
    width = 0.8 / n
    ticks, labels = [], []
    for i, p in enumerate(PARTY):
        colour = PARTY_COLOURS[p]
        for k, (label, values) in enumerate(series.items()):
            pos = i + (k - (n - 1) / 2) * width
            box = ax.boxplot([values[p]], positions=[pos], widths=width * 0.85, patch_artist=True,
                             medianprops={"color": "black"}, flierprops={"markersize": 3})["boxes"][0]
            if label == hatched:
                box.set(facecolor=to_rgba(colour, 0.2), edgecolor=colour, hatch="//")
            else:
                box.set(facecolor=colour, edgecolor="black")
            ticks.append(pos)
            labels.append(label)
    ax.set_xticks(ticks, labels, rotation=90, fontsize=8)
    for i, p in enumerate(PARTY):
        ax.text(i, -0.30, p, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=11, fontweight="bold", color=INK)
    ax.set_xlim(-0.6, len(PARTY) - 0.4)


def plot_spread(shares, fp22):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=SURFACE)
    style(ax)
    grouped_boxes(ax, {"2022": fp22, **shares}, hatched="2022")
    ax.set_ylabel("first preference share", color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(handles=[
        Patch(facecolor=to_rgba("grey", 0.2), edgecolor="grey", hatch="//", label="2022 election"),
        Patch(facecolor="grey", edgecolor="black", label="IPF modelled (seed labelled below box)"),
    ], loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False)
    ax.set_title("First preference share by district: 2022 v each seed", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{PREFIX}_spread.png", dpi=160, facecolor=SURFACE)


def plot_swing(shares, fp22, swing):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=SURFACE)
    style(ax)
    grouped_boxes(ax, {name: s - fp22 for name, s in shares.items()})
    for i, p in enumerate(PARTY):
        ax.hlines(swing[p], i - 0.45, i + 0.45, colors=INK, linestyles="--", lw=1.2, zorder=3)
    ax.axhline(0, color=INK_3, lw=0.8)
    ax.set_ylabel("modelled − 2022 share", color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(handles=[
        Patch(facecolor="grey", edgecolor="black", label="district swing (seed labelled below box)"),
        Line2D([], [], color=INK, ls="--", label="statewide survey swing"),
    ], loc="upper left", frameon=False)
    ax.set_title("Swing by district: modelled − 2022 lower house result, per seed", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{PREFIX}_swing.png", dpi=160, facecolor=SURFACE)


def plot_swing_by_region(shares, fp22, swing, region):
    regions = sorted(region.unique())
    fig, axes = plt.subplots(1, len(PARTY), figsize=(18, 5), sharey=True, facecolor=SURFACE)
    offsets = np.linspace(-0.27, 0.27, len(SEEDS))
    for ax, p in zip(axes, PARTY):
        style(ax)
        for k, (name, s) in enumerate(shares.items()):
            mean = (s[p] - fp22[p]).groupby(region.reindex(s.index)).mean().reindex(regions)
            ax.scatter(mean, np.arange(len(regions)) + offsets[k], s=36, marker=SEED_MARKERS[name],
                       color=SEED_COLOURS[name], edgecolor=SURFACE, linewidth=1, zorder=3,
                       label=name)
        ax.axvline(swing[p], color=INK, ls="--", lw=1)
        ax.axvline(0, color=INK_3, lw=0.8)
        ax.set_title(p, color=INK, fontsize=11)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_xlabel("average district swing", color=INK_2)
    axes[0].set_yticks(range(len(regions)), regions)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK, ls="--"))
    labels.append("statewide survey swing")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Average swing by upper house region, per seed", x=0.01, ha="left", y=1.06,
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{PREFIX}_swing_by_region.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")


def plot_lodo_seed(lodo, errors, name):
    data = lodo[lodo["seed"] == name]
    demogs = list(dict.fromkeys(data["demographic"]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor=SURFACE)
    lo, hi = 0, max(data["target"].max(), data["predicted"].max()) + 3
    for ax, demog in zip(axes.flat, demogs):
        style(ax)
        d = data[data["demographic"] == demog]
        categories = list(dict.fromkeys(d["category"]))
        for c, marker in zip(categories, CATEGORY_MARKERS):
            dc = d[d["category"] == c]
            ax.scatter(dc["target"], dc["predicted"], c=dc["party"].map(PARTY_COLOURS), marker=marker,
                       s=60, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.plot([lo, hi], [lo, hi], ls="--", color=INK_3, lw=0.8)
        ax.set(xlim=(lo, hi), ylim=(lo, hi))
        ax.grid(color=GRID, lw=0.8)
        e = errors[(errors["seed"] == name) & (errors["demographic"] == demog)].iloc[0]
        ax.set_title(f"{demog.title()} omitted — error {e['model_err']:.1f}pp (naive {e['naive_err']:.1f}pp)",
                     loc="left", color=INK, fontsize=10)
        ax.legend(handles=[Line2D([], [], marker=m, ls="", color=INK_2, label=c)
                           for c, m in zip(categories, CATEGORY_MARKERS)],
                  loc="upper left", frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("survey share (fitted to census + topline), %", color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("model's implied share, %", color=INK_2)
    fig.legend(handles=[Patch(color=c, label=p) for p, c in PARTY_COLOURS.items()],
               loc="upper center", ncol=len(PARTY), frameon=False, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(f"Leave one demographic out — {name} seed. On the dashed line = the model recovers "
                 f"how that group votes without being told.", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / f"{PREFIX}_lodo_{name.replace(' ', '_')}.png", dpi=160, facecolor=SURFACE)


def plot_lodo_summary(errors):
    demogs = list(dict.fromkeys(errors["demographic"]))
    groups = demogs + ["mean"]
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    style(ax)
    width = 0.8 / len(SEEDS)
    for k, name in enumerate(SEEDS):
        e = errors[errors["seed"] == name].set_index("demographic")["model_err"]
        values = [*e.reindex(demogs), e.mean()]
        x = np.arange(len(groups)) + (k - (len(SEEDS) - 1) / 2) * width
        ax.bar(x, values, width=width - 0.02, color=SEED_COLOURS[name], label=name, zorder=2)
    naive = errors.groupby("demographic", sort=False)["naive_err"].first()
    naive_values = [*naive.reindex(demogs), naive.mean()]
    ax.hlines(naive_values, np.arange(len(groups)) - 0.42, np.arange(len(groups)) + 0.42,
              colors=INK, linestyles="--", lw=1.2, zorder=3, label="naive (topline for every group)")
    ax.set_xticks(range(len(groups)), [g.title() for g in groups])
    ax.set_ylabel("mean absolute error, pp", color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_title("Leave one demographic out: error in the omitted group's party shares", loc="left",
                 color=INK, fontsize=12, pad=34)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{PREFIX}_lodo_summary.png", dpi=160, facecolor=SURFACE)


def main():
    shares, lodo, swing, region = run_all()
    districts = next(iter(shares.values())).index
    fp22 = load_first_preference_shares().reindex(districts)[PARTY]
    uh22 = load_upper_house_shares().reindex(districts)[PARTY]

    pd.concat(shares, names=["seed", "district"]).to_csv(OUT_DIR / f"{PREFIX}_district_shares.csv")
    spread = spread_table(shares, fp22)
    spread["onp_corr_upper_house_2022"] = [
        shares[r.seed]["One Nation"].corr(uh22["One Nation"]) if r.seed in shares and r.party == "One Nation"
        else np.nan for r in spread.itertuples()
    ]
    spread.to_csv(OUT_DIR / f"{PREFIX}_spread.csv", index=False)
    swings = swing_table(shares, fp22)
    swings.to_csv(OUT_DIR / f"{PREFIX}_swing.csv", index=False)
    lodo.to_csv(OUT_DIR / f"{PREFIX}_lodo.csv", index=False)
    errors = lodo_errors(lodo)
    errors.to_csv(OUT_DIR / f"{PREFIX}_lodo_errors.csv", index=False)

    pd.set_option("display.width", 200)
    print(spread.round(3).to_string())
    print(swings.round(2).to_string())
    print(errors.pivot(index="demographic", columns="seed", values="model_err").round(2).to_string())
    print(errors.groupby("seed", sort=False)[["model_err", "naive_err"]].mean().round(2).to_string())

    plot_spread(shares, fp22)
    plot_swing(shares, fp22, swing)
    plot_swing_by_region(shares, fp22, swing, region)
    for name in SEEDS:
        plot_lodo_seed(lodo, errors, name)
    plot_lodo_summary(errors)


if __name__ == "__main__":
    main()
