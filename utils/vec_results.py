"""Scrape 2022 Victorian state election first-preference results from the VEC.

District pages are discovered from the VEC's 2022 results page. Narracan's 2022
poll was failed (candidate death) and is taken from the January 2023
supplementary election, in which Labor did not stand a candidate.

Run as a script to write:
    data/vec_2022_first_preferences_candidates.csv  (one row per candidate)
    data/vec_2022_first_preference_shares.csv       (one row per district)
    data/vec_2022_two_candidate_preferred.csv       (two rows per district)
"""
import re
import time
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

BASE_URL = "https://www.vec.vic.gov.au"
INDEX_URL = f"{BASE_URL}/se22"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PARTY_GROUPS = {
    "Australian Labor Party - Victorian Branch": "Labor",
    "Liberal": "L/NP",
    "The Nationals": "L/NP",
    "Australian Greens": "Greens",
    "Pauline Hanson's One Nation": "One Nation",
}
# Everything else (including independents and Labour DLP) is 'Others'.
GROUPS = ["Labor", "L/NP", "Greens", "One Nation", "Others"]

_LINK = re.compile(
    r'<a href="(/results/state-election-results/'
    r'(?:2022-state-election-results/results-by-district/[a-z-]+-district-results'
    r'|state-by-elections-timeline/narracan-district-supplementary-election-results))"'
    r'[^>]*>\s*([^<]+?)\s+District'
)


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def get_district_links():
    """{district name: page path} for all districts linked from the 2022 results page."""
    return dict((name, path) for path, name in _LINK.findall(_get(INDEX_URL)))


def get_district_first_preferences(path):
    """First-preference table for one district page."""
    tables = pd.read_html(StringIO(_get(BASE_URL + path)))
    df = next(t for t in tables if "1st pref. votes" in t.columns)
    return df[["Candidate", "Party", "1st pref. votes"]].rename(
        columns={"Candidate": "candidate", "Party": "party", "1st pref. votes": "votes"}
    )


def scrape_candidates(delay=0.5):
    """One row per candidate in every district."""
    frames = []
    for district, path in sorted(get_district_links().items()):
        print(f"Fetching {district}...")
        df = get_district_first_preferences(path)
        df.insert(0, "district", district)
        df["supplementary"] = "supplementary" in path
        frames.append(df)
        time.sleep(delay)
    out = pd.concat(frames, ignore_index=True)
    out["party"] = out["party"].fillna("Independent")
    out["group"] = out["party"].map(PARTY_GROUPS).fillna("Others")
    return out


def get_district_two_candidate_preferred(path):
    """Final distribution table for one district page.

    Usually two candidates, but in a few seats the VEC stopped distributing once
    someone passed 50%, so the table has three.
    """
    tables = pd.read_html(StringIO(_get(BASE_URL + path)))
    # the column is labelled differently on some pages
    for col in ("Votes after distribution", "Preferred votes"):
        df = next((t for t in tables if col in t.columns), None)
        if df is not None:
            return df[["Candidate", "Party", col]].rename(
                columns={"Candidate": "candidate", "Party": "party", col: "votes"}
            )
    raise ValueError(f"no final distribution table on {path}: {[t.columns.tolist() for t in tables]}")


def scrape_two_candidate_preferred(delay=0.5):
    """Two rows per district (the final two candidates), with the winner flagged."""
    frames = []
    for district, path in sorted(get_district_links().items()):
        print(f"Fetching {district}...")
        df = get_district_two_candidate_preferred(path)
        df.insert(0, "district", district)
        df["supplementary"] = "supplementary" in path
        frames.append(df)
        time.sleep(delay)
    out = pd.concat(frames, ignore_index=True)
    out["party"] = out["party"].fillna("Independent")
    out["group"] = out["party"].map(PARTY_GROUPS).fillna("Others")
    out["share"] = out["votes"] / out.groupby("district")["votes"].transform("sum")
    out["elected"] = out["votes"] == out.groupby("district")["votes"].transform("max")
    return out


def load_two_candidate_preferred():
    """Saved 2CP results, one row per district (the winner), indexed by UPPER CASE district.

    Columns: winner / winner_party / winner_group / winner_share, and the same for runner_up.
    """
    tcp = pd.read_csv(DATA_DIR / "vec_2022_two_candidate_preferred.csv")
    tcp["district"] = tcp["district"].str.upper()
    cols = ["district", "candidate", "party", "group", "share"]
    win = tcp[tcp["elected"]][cols].set_index("district").add_prefix("winner_")
    # some seats have 3+ candidates left in the final count: keep the best placed loser
    lose = (
        tcp[~tcp["elected"]].sort_values("votes", ascending=False).drop_duplicates("district")
        [cols].set_index("district").add_prefix("runner_up_")
    )
    out = win.join(lose).rename(columns={"winner_candidate": "winner",
                                         "runner_up_candidate": "runner_up"})
    # True where the final count is a real 2CP (shares are then comparable across seats)
    out["full_2cp"] = tcp.groupby("district").size().reindex(out.index) == 2
    # winner's own first preference share (Lib and Nat can both stand, so not the group share)
    fp = pd.read_csv(DATA_DIR / "vec_2022_first_preferences_candidates.csv")
    fp["district"] = fp["district"].str.upper()
    fp["share"] = fp["votes"] / fp.groupby("district")["votes"].transform("sum")
    out["winner_fp_share"] = fp.set_index(["district", "candidate"])["share"].reindex(
        list(zip(out.index, out["winner"]))
    ).to_numpy()
    return out


def load_first_preference_shares():
    """Saved district shares, indexed by UPPER CASE district to match the 2022 boundaries."""
    shares = pd.read_csv(DATA_DIR / "vec_2022_first_preference_shares.csv")
    return shares.assign(district=shares["district"].str.upper()).set_index("district")


def first_preference_shares(candidates):
    """Share of formal first-preference votes by party group, one row per district."""
    votes = candidates.pivot_table(
        index="district", columns="group", values="votes", aggfunc="sum", fill_value=0
    ).reindex(columns=GROUPS, fill_value=0)
    shares = votes.div(votes.sum(axis=1), axis=0)
    shares["formal_votes"] = votes.sum(axis=1)
    shares["supplementary"] = candidates.groupby("district")["supplementary"].first()
    return shares


if __name__ == "__main__":
    candidates = scrape_candidates()
    candidates.to_csv(DATA_DIR / "vec_2022_first_preferences_candidates.csv", index=False)
    shares = first_preference_shares(candidates)
    shares.to_csv(DATA_DIR / "vec_2022_first_preference_shares.csv")
    print(f"{len(shares)} districts written")
    print("Parties grouped as Others:")
    print(candidates.query("group == 'Others'")["party"].value_counts().to_string())
    scrape_two_candidate_preferred().to_csv(DATA_DIR / "vec_2022_two_candidate_preferred.csv", index=False)
