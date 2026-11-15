"""Effect of applying the swing and of the smoothing strength alpha on the 2022 seeds.

For each 2022 seed (lower house, upper house, combined at upper house weight 0.5), with and
without the survey's statewide swing, and for alpha = m x the default (the mean district
formal vote, about 41.5k pseudo-votes), refit the full IPF and measure what
seed_comparison.py measures: spread across districts v 2022, swing size, leave one
demographic out (LODO) error and One Nation's ranking. all-ones is fitted once as a
reference.

m is the prior's weight relative to a district's own vote: a district keeps 1 / (1 + m) of
its own 2022 result and takes m / (1 + m) from the statewide shares. m = 1 is the default.

Run from the repo root:
    uv run python claudes_analyses/swing_alpha_comparison.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed_comparison import (  # noqa: E402
    CONVERGENCE_RATE, GRID, INK, INK_2, INK_3, PARTY, PARTY_COLOURS, SURFACE,
    demog_party_shares, fit, get_estimated_shares, get_seed_matrix, get_swing,
    grouped_boxes, setup, style,
)
from utils.vec_results import load_first_preference_shares  # noqa: E402
from utils.vec_upper_house import load_upper_house_shares  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PREFIX = "swing_alpha"

METHODS = {"lower house": ("2022-results", 0.0), "upper house": ("2022-upper-house", 1.0),
           "combined": ("2022-combined", 0.5)}
MULTIPLES = [0.01, 0.03, 0.1, 0.25, 0.5, 1, 2, 4, 10, 30]
DEFAULT_M = 1
# colour = seed; dashed = no swing. Palette slots 1-3 (dataviz skill).
SEED_COLOURS = {"lower house": "#eb6834", "upper house": "#1baf7a", "combined": "#2a78d6"}


def seed_array(method, upper_weight, swing, alpha, shape, districts):
    return get_seed_matrix(method, shape, districts, weight=1.0, upper_house_weight=upper_weight,
                           swing=swing, alpha=alpha)


def seed_shares(seed):
    """District x party shares inside the seed (the district pattern IPF starts from)."""
    s = seed[(slice(None), *[0] * (seed.ndim - 2), slice(None))]
    return pd.DataFrame(s / s.sum(axis=1, keepdims=True), columns=PARTY)


def run_all():
    population, demogs, topline, topline_p, ipf_mats, region = setup()
    districts = list(population.index)
    shape = (len(districts), *[len(d.get_columns()) for d in demogs], len(PARTY))
    swing = get_swing(topline)
    mean_votes = load_first_preference_shares()["formal_votes"].mean()

    configs = [("all-ones", False, np.nan)]
    configs += [(s, sw, m) for s in METHODS for sw in (False, True) for m in MULTIPLES]

    shares, seeds, lodo, fitq = {}, {}, [], []
    for n, (name, use_swing, m) in enumerate(configs, start=1):
        if name == "all-ones":
            seed = get_seed_matrix("all-ones", shape)
        else:
            method, uw = METHODS[name]
            seed = seed_array(method, uw, swing if use_swing else None, m * mean_votes, shape, districts)
        key = (name, use_swing, m)
        X = fit(seed, demogs, ipf_mats)
        shares[key] = get_estimated_shares(X, demogs, population.index)[PARTY]
        seeds[key] = seed_shares(seed).set_index(population.index)

        # worst relative miss on any demographic x party margin: non-zero means the seed's zeros
        # stopped the fit reaching the survey tables
        rel = [np.abs(X.sum(axis=tuple(a for a in range(X.ndim) if a not in (i, X.ndim - 1))) / ipf_mats[d.name] - 1).max()
               for i, d in enumerate(demogs, start=1)]
        fitq.append({"seed": name, "swing": use_swing, "m": m, "max_margin_miss_pct": 100 * max(rel)})

        for ax, demog in enumerate(demogs, start=1):
            predicted = demog_party_shares(fit(seed, demogs, ipf_mats, omit=demog.name), ax)
            target = ipf_mats[demog.name] / ipf_mats[demog.name].sum(axis=1, keepdims=True)
            for i, category in enumerate(demog.get_columns()):
                for j, party in enumerate(PARTY):
                    lodo.append({"seed": name, "swing": use_swing, "m": m, "demographic": demog.name,
                                 "category": category, "party": party,
                                 "target": 100 * target[i, j], "predicted": 100 * predicted[i, j]})
        print(f"[{n}/{len(configs)}] {key}", flush=True)

    naive = pd.Series(100 * topline_p / topline_p.sum(), index=PARTY)
    lodo = pd.DataFrame(lodo)
    lodo["naive"] = lodo["party"].map(naive)
    return shares, seeds, lodo, pd.DataFrame(fitq), swing, region


def summarise(shares, seeds, lodo, fitq, fp22, uh22):
    rows = []
    for (name, use_swing, m), s in shares.items():
        sd = seeds[(name, use_swing, m)]
        change = 100 * (s - fp22)
        for p in PARTY:
            rows.append({"seed": name, "swing": use_swing, "m": m, "party": p,
                         "sd": 100 * s[p].std(), "rank_corr_2022": s[p].corr(fp22[p], method="spearman"),
                         "swing_sd": change[p].std(), "mean_swing": change[p].mean(),
                         "min_swing": change[p].min(), "max_swing": change[p].max(),
                         "seed_sd": 100 * sd[p].std(), "min_share": 100 * s[p].min(), "max_share": 100 * s[p].max()})
    summary = pd.DataFrame(rows)

    err = lodo.assign(model_err=(lodo["predicted"] - lodo["target"]).abs(),
                      naive_err=(lodo["naive"] - lodo["target"]).abs())
    lodo_err = err.groupby(["seed", "swing", "m", "demographic"], sort=False, dropna=False)[["model_err", "naive_err"]].mean().reset_index()

    onp = []
    top_uh = set(uh22["One Nation"].nlargest(10).index)
    for (name, use_swing, m), s in shares.items():
        top = s["One Nation"].nlargest(10)
        onp.append({"seed": name, "swing": use_swing, "m": m, "top10_overlap_uh2022": len(top_uh & set(top.index)),
                    "corr_uh2022": s["One Nation"].corr(uh22["One Nation"]),
                    "contested5_in_top10": len({"MORWELL", "NARRACAN", "BENDIGO EAST", "MACEDON", "PAKENHAM"} & set(top.index)),
                    "top10_range_pp": 100 * (top.max() - top.min()), "top10": ", ".join(f"{d} {100 * v:.1f}" for d, v in top.items())})
    return summary, lodo_err, pd.DataFrame(onp)


# --- plots ---

def line_panels(summary, value, ylabel, title, fname, ref=None, ref_label=None, parties=PARTY):
    """One panel per party: ``value`` against alpha multiple, colour = seed, dashed = no swing."""
    fig, axes = plt.subplots(1, len(parties), figsize=(18, 4.3), sharex=True, facecolor=SURFACE)
    for ax, p in zip(axes, parties):
        style(ax)
        d = summary[(summary["party"] == p) & (summary["seed"] != "all-ones")]
        for (name, use_swing), g in d.groupby(["seed", "swing"]):
            ax.plot(g["m"], g[value], color=SEED_COLOURS[name], ls="-" if use_swing else "--", lw=1.8,
                    marker="o" if use_swing else "s", ms=3.5)
        flat = summary[(summary["party"] == p) & (summary["seed"] == "all-ones")]
        if len(flat):
            ax.axhline(flat[value].iloc[0], color=INK_3, lw=1, ls=":")
        if ref is not None:
            ax.axhline(ref[p], color=INK, lw=1.2, ls="-.")
        ax.axvline(DEFAULT_M, color=INK_3, lw=0.8)
        ax.set_xscale("log")
        ax.set_title(p, color=PARTY_COLOURS[p] if p != "L/NP" else "blue", fontsize=11)
        ax.set_xlabel("alpha (x default)", color=INK_2)
        ax.grid(color=GRID, lw=0.8)
    axes[0].set_ylabel(ylabel, color=INK_2)
    handles = [Line2D([], [], color=c, label=n) for n, c in SEED_COLOURS.items()]
    handles += [Line2D([], [], color=INK_2, ls="-", marker="o", ms=3.5, label="swing applied"),
                Line2D([], [], color=INK_2, ls="--", marker="s", ms=3.5, label="no swing"),
                Line2D([], [], color=INK_3, ls=":", label="all-ones")]
    if ref_label:
        handles.append(Line2D([], [], color=INK, ls="-.", label=ref_label))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, x=0.01, ha="left", y=1.1, color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{PREFIX}_{fname}.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_lodo(lodo_err):
    mean = lodo_err.groupby(["seed", "swing", "m"], sort=False, dropna=False)[["model_err", "naive_err"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    style(ax)
    for (name, use_swing), g in mean[mean["seed"] != "all-ones"].groupby(["seed", "swing"]):
        ax.plot(g["m"], g["model_err"], color=SEED_COLOURS[name], ls="-" if use_swing else "--", lw=1.8,
                marker="o" if use_swing else "s", ms=3.5)
    flat = mean[mean["seed"] == "all-ones"].iloc[0]
    ax.axhline(flat["model_err"], color=INK_3, ls=":", lw=1.2)
    ax.axhline(flat["naive_err"], color=INK, ls="-.", lw=1.2)
    ax.axvline(DEFAULT_M, color=INK_3, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (x default)", color=INK_2)
    ax.set_ylabel("mean absolute error, pp", color=INK_2)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(handles=[Line2D([], [], color=c, label=n) for n, c in SEED_COLOURS.items()] + [
        Line2D([], [], color=INK_2, ls="-", marker="o", ms=3.5, label="swing applied"),
        Line2D([], [], color=INK_2, ls="--", marker="s", ms=3.5, label="no swing"),
        Line2D([], [], color=INK_3, ls=":", label="all-ones"),
        Line2D([], [], color=INK, ls="-.", label="naive (topline)")], frameon=False, ncol=2, fontsize=9)
    ax.set_title("Leave one demographic out: mean error against alpha", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{PREFIX}_lodo.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_onp(onp):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), sharex=True, facecolor=SURFACE)
    for ax, (col, label) in zip(axes, [("top10_overlap_uh2022", "seats shared with actual 2022 upper house top 10"),
                                       ("corr_uh2022", "correlation with 2022 upper house One Nation vote"),
                                       ("contested5_in_top10", "of the 5 contested lower house seats, number in top 10")]):
        style(ax)
        d = onp[onp["seed"] != "all-ones"]
        for (name, use_swing), g in d.groupby(["seed", "swing"]):
            ax.plot(g["m"], g[col], color=SEED_COLOURS[name], ls="-" if use_swing else "--", lw=1.8,
                    marker="o" if use_swing else "s", ms=3.5)
        ax.axhline(onp[onp["seed"] == "all-ones"][col].iloc[0], color=INK_3, ls=":", lw=1.2)
        ax.axvline(DEFAULT_M, color=INK_3, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("alpha (x default)", color=INK_2)
        ax.set_title(label, loc="left", color=INK, fontsize=10)
        ax.grid(color=GRID, lw=0.8)
    axes[0].legend(handles=[Line2D([], [], color=c, label=n) for n, c in SEED_COLOURS.items()] + [
        Line2D([], [], color=INK_2, ls="-", marker="o", ms=3.5, label="swing applied"),
        Line2D([], [], color=INK_2, ls="--", marker="s", ms=3.5, label="no swing"),
        Line2D([], [], color=INK_3, ls=":", label="all-ones")], frameon=False, fontsize=8, loc="best")
    fig.suptitle("One Nation: where the model puts it, against alpha", x=0.01, ha="left", y=1.02, color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{PREFIX}_one_nation.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_boxes(shares, fp22, use_swing, fname, title, name="combined"):
    series = {"2022": fp22}
    for m in (0.03, 0.25, 1, 4, 30):
        series[f"{m:g}x"] = shares[(name, use_swing, m)]
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=SURFACE)
    style(ax)
    grouped_boxes(ax, series, hatched="2022")
    ax.set_ylabel("first preference share", color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{PREFIX}_{fname}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    shares, seeds, lodo, fitq, swing, region = run_all()
    districts = next(iter(shares.values())).index
    fp22 = load_first_preference_shares().reindex(districts)[PARTY]
    uh22 = load_upper_house_shares().reindex(districts)[PARTY]

    summary, lodo_err, onp = summarise(shares, seeds, lodo, fitq, fp22, uh22)
    pd.concat({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in shares.items()}, names=["config", "district"]) \
        .to_csv(OUT_DIR / f"{PREFIX}_district_shares.csv")
    summary.to_csv(OUT_DIR / f"{PREFIX}_summary.csv", index=False)
    lodo_err.to_csv(OUT_DIR / f"{PREFIX}_lodo_errors.csv", index=False)
    onp.to_csv(OUT_DIR / f"{PREFIX}_one_nation.csv", index=False)
    fitq.to_csv(OUT_DIR / f"{PREFIX}_fit_quality.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500, "display.max_colwidth", 120)
    for value in ("sd", "rank_corr_2022", "swing_sd", "seed_sd"):
        print(f"\n== {value} ==")
        print(summary.pivot_table(index=["seed", "swing", "m"], columns="party", values=value, sort=False)[PARTY].round(2).to_string())
    print("\n== LODO mean error ==")
    print(lodo_err.groupby(["seed", "swing", "m"], sort=False, dropna=False)[["model_err", "naive_err"]].mean().round(3).to_string())
    print("\n== One Nation ==")
    print(onp.drop(columns="top10").round(3).to_string())
    print("\n== fit quality ==")
    print(fitq.round(4).to_string())

    ref_sd = {p: 100 * fp22[p].std() for p in PARTY}
    line_panels(summary, "sd", "SD across districts, pp", "Spread across districts against alpha (dash-dot = 2022 actual)",
                "spread", ref=ref_sd, ref_label="2022 actual")
    line_panels(summary, "rank_corr_2022", "rank correlation with 2022",
                "Rank correlation with the 2022 lower house result against alpha", "rank_corr")
    line_panels(summary, "swing_sd", "SD of district swing, pp", "SD of district swings (modelled − 2022) against alpha", "swing_sd")
    line_panels(summary, "mean_swing", "mean district swing, pp", "Mean district swing (modelled − 2022) against alpha; dash-dot = survey swing",
                "swing_mean", ref={p: 100 * swing[p] for p in PARTY}, ref_label="survey swing")
    plot_lodo(lodo_err)
    plot_onp(onp)
    plot_boxes(shares, fp22, True, "boxes_swing", "Combined seed, swing applied: district shares by alpha (x default)")
    plot_boxes(shares, fp22, False, "boxes_noswing", "Combined seed, no swing: district shares by alpha (x default)")


if __name__ == "__main__":
    main()
