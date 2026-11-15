"""Boundaries and the 2021 SED -> 2022 district crosswalk.

Census data is published on 2021 ABS State Electoral Divisions (SEDs), which
predate the 2022 Victorian redistribution. The crosswalk splits each SED's
counts across the 2022 districts it overlaps, in proportion to overlap area.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

SPATIAL_DIR = Path(__file__).resolve().parent.parent / "data/spatial"
SED_2021_PATH = SPATIAL_DIR / "SED_2021_AUST_GDA2020_SHP/SED_2021_AUST_GDA2020.shp"
DISTRICT_2022_PATH = (
    SPATIAL_DIR / "ll_gda2020/esrishape/whole_of_dataset/victoria/VMADMIN/STATE_ASSEMBLY_2022.shp"
)
# GDA2020 / Vicgrid, units in metres
AREA_CRS = 7899


def load_2021_seds():
    """Victorian 2021 SEDs, keyed on SED_CODE21 (e.g. '20106')."""
    return (
        gpd.read_file(SED_2021_PATH)
        .query("STE_NAME21 == 'Victoria'")
        .loc[:, ["SED_CODE21", "SED_NAME21", "geometry"]]
        .reset_index(drop=True)
    )


def load_2022_districts():
    """The 88 Victorian 2022 Legislative Assembly districts, keyed on DISTRICT (UPPER CASE)."""
    return gpd.read_file(DISTRICT_2022_PATH).loc[:, ["DISTRICT", "REGION", "geometry"]]


def calculate_2021_to_2022_mapping(seds_2021, districts_2022, crs=AREA_CRS):
    """Share of each 2021 SED's population allocated to each 2022 district.

    Census values are counts, so the weight is overlap_area / area of the 2021
    SED (assumes uniform density within a SED). Weights are renormalised to sum
    to 1 per SED, which absorbs slivers where the coastlines differ.

    TODO: SA1-level population weights would handle rural/urban boundary moves
    better than area.
    """
    seds = seds_2021.to_crs(crs)
    seds["area_2021"] = seds.geometry.area
    intersection = gpd.overlay(
        districts_2022.to_crs(crs), seds, how="intersection", keep_geom_type=False
    )
    intersection["weight"] = intersection.geometry.area / intersection["area_2021"]
    intersection["weight"] /= intersection.groupby("SED_CODE21")["weight"].transform("sum")
    return pd.DataFrame(intersection.loc[:, ["SED_CODE21", "DISTRICT", "weight"]])


def map_2021_census_to_2022_districts(mapping, census_df):
    """Allocate per-SED census counts (indexed by 'SED20106'-style codes) to 2022 districts."""
    counts = census_df.copy()
    counts.index = counts.index.str.removeprefix("SED")
    columns = list(counts.columns)
    merged = mapping.merge(counts, left_on="SED_CODE21", right_index=True, how="inner")
    merged[columns] = merged[columns].mul(merged["weight"], axis=0)
    return merged.groupby("DISTRICT")[columns].sum()
