"""Scrape 2022 Victorian Legislative Council (upper house) first preferences by district.

Upper house votes are counted by lower house district: each region's "Results by
Voting Centre" spreadsheet has one sheet per district, ending in a "<name> District
Total" block with each group's ATL + BTL first-preference votes. Narracan's upper
house poll went ahead as normal in 2022 (only the lower house poll failed).

Run as a script to write:
    data/vec_2022_upper_house_groups.csv  (one row per district x group)
    data/vec_2022_upper_house_shares.csv  (one row per district)
"""
import re
import time
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd

from utils.vec_results import GROUPS, PARTY_GROUPS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_URL = "https://itsitecoreblobvecprd01.blob.core.windows.net/public-files/State/Reports"
REGIONS = [
    "Eastern Victoria",
    "North-Eastern Metropolitan",
    "Northern Metropolitan",
    "Northern Victoria",
    "South-Eastern Metropolitan",
    "Southern Metropolitan",
    "Western Metropolitan",
    "Western Victoria",
]
# The Coalition ran joint tickets in the country regions.
UPPER_HOUSE_PARTY_GROUPS = {**PARTY_GROUPS, "Liberal / The Nationals": "L/NP"}

_GROUP =re.compile(r"^GROUP [A-Z]+ - (.+)$")
_FORMAL = re.compile(r"^Formal Votes\s*:\s*([\d,]+)")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def region_url(region):
    return f"{REPORTS_URL}/{urllib.parse.quote(f'{region} Region-Results by Voting Centre.xls')}"


def _to_int(value):
    return int(str(value).replace(",", ""))


def parse_district_sheet(df):
    """(district, formal votes, {group label: votes}) from one district's sheet."""
    labels = df[0].fillna("").astype(str).str.strip()
    total_row = labels[labels.str.endswith("District Total")].index[0]
    district = labels[total_row].removesuffix(" District Total")
    formal = next(_to_int(m.group(1)) for m in labels.map(_FORMAL.match) if m)

    votes = {}
    for row in range(total_row + 1, len(df)):
        if labels[row] == "Informal":
            break
        match = _GROUP.match(labels[row])
        # Ungrouped candidates are listed by name rather than as a GROUP.
        party = match.group(1) if match else "Ungrouped"
        votes[party] = votes.get(party, 0) + _to_int(df.iloc[row, 4])
    if sum(votes.values()) != formal:
        raise ValueError(f"{district}: group votes sum to {sum(votes.values())}, formal votes are {formal}")
    return district, formal, votes


def scrape_groups(delay=0.5):
    """One row per district x upper house group."""
    rows = []
    for region in REGIONS:
        print(f"Fetching {region} Region...")
        sheets = pd.read_excel(BytesIO(_get(region_url(region))), sheet_name=None, header=None, engine="calamine")
        for df in sheets.values():
            district, formal, votes = parse_district_sheet(df)
            rows += [
                {"region": region, "district": district, "party": party, "votes": n}
                for party, n in votes.items()
            ]
        time.sleep(delay)
    out = pd.DataFrame(rows)
    out["group"] = out["party"].map(UPPER_HOUSE_PARTY_GROUPS).fillna("Others")
    return out


def upper_house_shares(groups):
    """Share of formal upper house first-preference votes by party group, one row per district."""
    votes = groups.pivot_table(
        index="district", columns="group", values="votes", aggfunc="sum", fill_value=0
    ).reindex(columns=GROUPS, fill_value=0)
    shares = votes.div(votes.sum(axis=1), axis=0)
    shares["formal_votes"] = votes.sum(axis=1)
    shares["region"] = groups.groupby("district")["region"].first()
    return shares


def load_upper_house_shares():
    """Saved district shares, indexed by UPPER CASE district to match the 2022 boundaries."""
    shares = pd.read_csv(DATA_DIR / "vec_2022_upper_house_shares.csv")
    return shares.assign(district=shares["district"].str.upper()).set_index("district")


if __name__ == "__main__":
    groups = scrape_groups()
    groups.to_csv(DATA_DIR / "vec_2022_upper_house_groups.csv", index=False)
    shares = upper_house_shares(groups)
    shares.to_csv(DATA_DIR / "vec_2022_upper_house_shares.csv")
    print(f"{len(shares)} districts written")
    print("Parties grouped as Others:")
    print(groups.query("group == 'Others'").groupby("party")["votes"].sum().sort_values(ascending=False).to_string())
