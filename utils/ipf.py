"""IPF estimation of party vote by demographic for the 2022 Victorian districts.

The fitted array has axes [district, *demographics, party]. Its margins are
fitted to (a) census district x demographic counts and (b) demographic x party
counts from a survey, each first fitted to the survey's topline party vote.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from ipfn import ipfn

from utils.census import (
    get_census_adult_citizen_population,
    get_census_adult_population,
    get_census_age_data,
    get_census_education_data,
    get_census_income_data,
    get_census_language_data,
    get_census_sex_data,
    get_census_tenure_data,
    rescale_to_population,
)
from utils.spatial import map_2021_census_to_2022_districts
from utils.vec_results import load_first_preference_shares

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEX = ['Male', 'Female']
AGE = ['18-34', '35-54', '55+']
LANGUAGE = ['Other Language at Home', 'English Only']
INCOME = ['<$45K', '$45-125K', '$125K+']
TENURE = ['Own Home Outright', 'Mortgage Holder', 'Renter']
EDUCATION = ['School Education', 'University Education', 'TAFE Education']
PARTY = ['Labor', 'Greens', 'Others', 'One Nation', 'L/NP']

# survey name -> (demographic breakdown CSV, row label in topline_primary_vote.csv)
SURVEYS = {
    "aug2026": (DATA_DIR / "demographic_primary_vote_aug2026.csv", "August 26"),
    "sep2026": (DATA_DIR / "demographic_primary_vote_sep2026.csv", "Sep 26"),
}
SEED_MATRIX_METHODS = ["all-ones", "2022-results"]
ZERO_SHARE_METHODS = ["epsilon"]
POPULATION_BASES = ["adult", "adult-citizen", "formal-votes-2022"]


class demogCat:

    def __init__(
            self,
            name: str,
            columns: list[str],
            data: pd.DataFrame,
            population: pd.Series | None = None,
        ):
        self.name = name
        self.columns = columns
        data = data[columns]
        if population is not None:
            data = rescale_to_population(data, population.reindex(data.index))
        self.data = data

    def get_totals(self) -> np.ndarray:
        return self.data.to_numpy().sum(axis=0)

    def get_columns(self) -> list[str]:
        return self.columns

    def get_demog_data(self) -> pd.DataFrame:
        return self.data[self.columns]


def load_survey(survey: str) -> tuple[pd.DataFrame, pd.Series]:
    """(demographic breakdown, topline party percentages) for a survey in SURVEYS."""
    if survey not in SURVEYS:
        raise ValueError(f"Unknown survey {survey!r}; choose from {list(SURVEYS)}")
    path, poll_label = SURVEYS[survey]
    demographics = pd.read_csv(path)
    topline = pd.read_csv(DATA_DIR / "topline_primary_vote.csv").set_index("Poll").loc[poll_label, PARTY]
    return demographics, topline


def calc_party_shares(demographics: pd.DataFrame, parties: list[str]) -> pd.DataFrame:
    pct = demographics.set_index('Demographic')[parties].to_numpy() / 100
    return pd.DataFrame(pct, index=demographics.Demographic, columns=parties)


# --- population base ---

def get_population_base(method: str, mapping: pd.DataFrame) -> pd.Series:
    """Population each 2022 district's marginals are rescaled to, indexed by DISTRICT.

    Every base sets district size only: each marginal is scaled by a single
    factor per district, so the demographic mix stays the census all-adult mix.
    Turnout and citizenship differences between demographics are not adjusted.
    """
    if method == "adult":
        return get_adult_population_base(mapping)
    elif method == "adult-citizen":
        return get_adult_citizen_population_base(mapping)
    elif method == "formal-votes-2022":
        return get_formal_votes_population_base()
    raise ValueError(f"Unknown population base {method!r}; choose from {POPULATION_BASES}")


def get_adult_population_base(mapping: pd.DataFrame) -> pd.Series:
    """Persons aged 18+ (G04), crosswalked to 2022 districts."""
    return map_2021_census_to_2022_districts(mapping, get_census_adult_population().to_frame()).iloc[:, 0]


def get_adult_citizen_population_base(mapping: pd.DataFrame) -> pd.Series:
    """Approximate citizens aged 18+ (G04 x G01 citizen rate), crosswalked to 2022 districts."""
    return map_2021_census_to_2022_districts(mapping, get_census_adult_citizen_population().to_frame()).iloc[:, 0]


def get_formal_votes_population_base() -> pd.Series:
    """Formal first-preference votes per district at the 2022 election."""
    return load_first_preference_shares()["formal_votes"].astype(float).rename("Formal Votes").rename_axis("DISTRICT")


def build_demogs(mapping: pd.DataFrame, population: pd.Series) -> list[demogCat]:
    """Census marginals crosswalked to 2022 districts and rescaled to ``population``."""
    sources = [
        ("education", EDUCATION, get_census_education_data),
        ("age", AGE, get_census_age_data),
        ("income", INCOME, get_census_income_data),
        ("language", LANGUAGE, get_census_language_data),
        ("sex", SEX, get_census_sex_data),
        ("tenure", TENURE, get_census_tenure_data),
    ]
    districts = population.sort_index().index
    return [
        demogCat(
            name=name,
            columns=columns,
            data=map_2021_census_to_2022_districts(mapping, get_data()).reindex(districts),
            population=population,
        )
        for name, columns, get_data in sources
    ]


# --- seed matrix ---

def get_seed_matrix(
        method: str,
        shape: tuple[int, ...],
        districts: list[str] | None = None,
        zero_method: str = "epsilon",
        **zero_kwargs,
    ) -> np.ndarray:
    """Initial array for the full IPF, axes [district, *demographics, party]."""
    if method == "all-ones":
        return get_all_ones_seed_mat(shape)
    elif method == "2022-results":
        return get_2022_results_seed_mat(shape, districts, zero_method, **zero_kwargs)
    raise ValueError(f"Unknown seed matrix {method!r}; choose from {SEED_MATRIX_METHODS}")


def get_all_ones_seed_mat(shape: tuple[int, ...]) -> np.ndarray:
    return np.ones(shape)


def get_2022_results_seed_mat(
        shape: tuple[int, ...],
        districts: list[str],
        zero_method: str = "epsilon",
        **zero_kwargs,
    ) -> np.ndarray:
    """2022 first-preference shares per district x party, constant across demographics."""
    shares = load_first_preference_shares().reindex(districts)[PARTY]
    if shares.isna().any().any():
        missing = shares.index[shares.isna().any(axis=1)].tolist()
        raise ValueError(f"No 2022 results for districts: {missing}")
    shares = handle_zero_shares(shares, zero_method, **zero_kwargs)
    district_party = shares.to_numpy().reshape(shape[0], *[1] * (len(shape) - 2), shape[-1])
    return np.broadcast_to(district_party, shape).copy()


def handle_zero_shares(shares: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
    """Remove 0.0 shares, which IPF can never scale up (e.g. parties that did not stand)."""
    if method == "epsilon":
        return apply_epsilon_floor(shares, **kwargs)
    raise ValueError(f"Unknown zero-share method {method!r}; choose from {ZERO_SHARE_METHODS}")


def apply_epsilon_floor(shares: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Floor shares at ``eps`` and renormalise each row to sum to 1."""
    floored = shares.clip(lower=eps)
    return floored.div(floored.sum(axis=1), axis=0)


# --- fitting ---

def run_ipf(
        original_matrix: np.ndarray,
        row_col_totals: list[np.ndarray],
        dimensions: list[list[int]],
        convergence_rate: float = 1.0e-10,
        max_iteration: int = 1000,
        ) -> np.ndarray:
    return ipfn.ipfn(
        original = original_matrix,
        aggregates = row_col_totals,
        dimensions = dimensions,
        convergence_rate=convergence_rate,
        max_iteration=max_iteration,
    ).iteration()


def fit_demog_party_margins(
        demogs: list[demogCat],
        pct: pd.DataFrame,
        topline_p: np.ndarray,
        convergence_rate: float = 1.0e-10,
    ) -> dict[str, np.ndarray]:
    """Per demographic: survey party shares fitted to census totals (rows) and topline (columns)."""
    return {
        demog.name: run_ipf(
            original_matrix=pct.loc[demog.get_columns(), PARTY].to_numpy(dtype=float, copy=True),
            row_col_totals=[demog.get_totals(), topline_p],
            dimensions=[[0], [1]],
            convergence_rate=convergence_rate,
        )
        for demog in demogs
    }


def fit_full(
        seed: np.ndarray,
        demogs: list[demogCat],
        ipf_mats: dict[str, np.ndarray],
        convergence_rate: float = 1.0e-10,
    ) -> np.ndarray:
    """Fit [district, *demographics, party] to district x demographic and demographic x party margins."""
    party_axis = len(demogs) + 1
    demog_axes = range(1, party_axis)
    return run_ipf(
        original_matrix=seed.astype(float, copy=True),
        row_col_totals=[
            *[demog.data.to_numpy() for demog in demogs],
            *[ipf_mats[demog.name] for demog in demogs],
        ],
        dimensions=[
            *[[0, ax] for ax in demog_axes],
            *[[ax, party_axis] for ax in demog_axes],
        ],
        convergence_rate=convergence_rate,
    )


def margin_errors(
        X: np.ndarray,
        demogs: list[demogCat],
        ipf_mats: dict[str, np.ndarray],
        topline_p: np.ndarray,
    ) -> pd.DataFrame:
    """Max absolute error of each fitted margin against its target."""
    party_axis = X.ndim - 1
    rows = []
    for i, demog in enumerate(demogs, start=1):
        m = ipf_mats[demog.name]
        other_axes = tuple(ax for ax in range(1, party_axis) if ax != i)
        rows.append({
            "demographic": demog.name,
            "2d_vs_census_totals": np.abs(m.sum(axis=1) - demog.get_totals()).max(),
            "2d_vs_topline": np.abs(m.sum(axis=0) - topline_p).max(),
            "full_vs_demog_party": np.abs(X.sum(axis=(0, *other_axes)) - m).max(),
            "full_vs_district_demog": np.abs(
                X.sum(axis=(*other_axes, party_axis)) - demog.data.to_numpy()
            ).max(),
        })
    return pd.DataFrame(rows).set_index("demographic")


def get_totals_by_region(X: np.ndarray, axes: tuple[int, ...], columns: list[str], index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(X.sum(axis=axes), index=index, columns=columns)


def get_shares_by_region(X: np.ndarray, axes: tuple[int, ...], columns: list[str], index: pd.Index) -> pd.DataFrame:
    totals = X.sum(axis=axes)
    return pd.DataFrame(totals / totals.sum(axis=1, keepdims=True), index=index, columns=columns)


def get_estimated_shares(X: np.ndarray, demogs: list[demogCat], index: pd.Index) -> pd.DataFrame:
    """Party shares then each demographic's shares, one row per district."""
    party_axis = X.ndim - 1
    frames = [get_shares_by_region(X, tuple(range(1, party_axis)), PARTY, index)]
    for i, demog in enumerate(demogs, start=1):
        axes = tuple(ax for ax in range(1, X.ndim) if ax != i)
        frames.append(get_shares_by_region(X, axes, demog.get_columns(), index))
    return pd.concat(frames, axis=1)
