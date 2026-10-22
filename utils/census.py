"""Per-SED census marginals for the demographics in data/demographic_primary_vote_*.csv.

Each ``get_census_*_data`` returns a DataFrame indexed by SED_CODE_2021 with one
column per poll demographic. Marginals are independent (not cross-tabulated).
Tables without an age breakdown are returned as raw counts on their own
population base; use ``rescale_to_population`` explicitly to align them to 18+.
"""
from pathlib import Path

import pandas as pd

from utils.mappings import (
    age18_mapping,
    education_mapping,
    income_brackets,
    language_mapping,
    sex_mapping,
    tenure_mapping,
)

CENSUS_DIR = (
    Path(__file__).resolve().parent.parent
    / "data/census/2021 Census GCP State Electroral Division for VIC"
)
# 'No usual address' and 'Migratory - Offshore - Shipping'
NON_GEOGRAPHIC_SEDS = {"SED29494", "SED29797"}


def _read_census_table(table, parts=""):
    """Read a GCP table (concatenating lettered parts, e.g. parts='AB') indexed by SED."""
    frames = [
        pd.read_csv(CENSUS_DIR / f"2021Census_{table}{part}_VIC_SED.csv", index_col="SED_CODE_2021")
        for part in (parts or [""])
    ]
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.drop(index=NON_GEOGRAPHIC_SEDS, errors="ignore")


def _aggregate(df, mapping):
    """Sum census columns into demographics.

    mapping: {census_col: demographic} or {census_col: {demographic: weight}}.
    Columns are ordered by first appearance in the mapping.
    """
    out = {}
    for col, target in mapping.items():
        weights = {target: 1.0} if isinstance(target, str) else target
        for demographic, weight in weights.items():
            out[demographic] = out.get(demographic, 0) + weight * df[col]
    return pd.DataFrame(out, index=df.index)


def get_census_adult_population():
    """Persons aged 18+ per SED (G04)."""
    return get_census_age_data().sum(axis=1).rename("Adult Population")


def get_census_sex_data():
    """Male / Female, persons aged 18+ (G04)."""
    return _aggregate(_read_census_table("G04", "AB"), sex_mapping)


def get_census_age_data():
    """18-34 / 35-54 / 55+ (G04)."""
    return _aggregate(_read_census_table("G04", "AB"), age18_mapping)


def get_census_education_data():
    """School / TAFE / University Education, persons aged 18+ (G49 + G04).

    TAFE and University come from G49 (persons 15+ with a non-school
    qualification; 15-17 year olds hold almost none). School Education is the
    18+ population minus everyone with a non-school qualification, including
    those whose level was not stated or inadequately described.
    """
    g49 = _read_census_table("G49", "AB")
    df = _aggregate(g49, education_mapping)
    df.insert(0, "School Education", get_census_adult_population() - g49["P_Tot_Total"])
    return df


def get_census_income_data():
    """<$45K / $45-125K / $125K+ personal income, persons aged 18+ (G17 + G04).

    G17's youngest band is 15-19, so the 15-17 share of it is removed using the
    SED's single-year age distribution from G04 (assumes 15-17 and 18-19 year
    olds in that band have the same income distribution).
    """
    g17 = _read_census_table("G17", "ABC")
    g04 = _read_census_table("G04", "AB")
    share_15_17 = 1 - (g04["Age_yr_18_P"] + g04["Age_yr_19_P"]) / g04["Age_yr_15_19_P"]
    adults = pd.DataFrame({
        bracket: g17[f"P_{bracket}_Tot"] - share_15_17 * g17[f"P_{bracket}_15_19_yrs"]
        for bracket in income_brackets
    })
    return _aggregate(adults, income_brackets)


def get_census_tenure_data():
    """Own Home Outright / Mortgage Holder / Renter, occupied private DWELLINGS (G37).

    Not on a persons-18+ basis; apply ``rescale_to_population``.
    """
    return _aggregate(_read_census_table("G37"), tenure_mapping)


def get_census_language_data():
    """English Only / Other Language at Home, persons of ALL AGES (G01).

    Not on a persons-18+ basis; apply ``rescale_to_population``.
    """
    return _aggregate(_read_census_table("G01"), language_mapping)


def rescale_to_population(df, population=None):
    """Scale each SED row of ``df`` proportionally so it sums to ``population``.

    Defaults to the 18+ population from ``get_census_adult_population``.
    """
    if population is None:
        population = get_census_adult_population()
    return df.mul(population / df.sum(axis=1), axis=0)
