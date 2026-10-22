gender_mapping = {
    "Tot_P_M": "Male",
    "Tot_P_F": "Female"
}

age_mapping = {
    "Age_0_4_yr_P": None,
    "Age_5_14_yr_P": None,
    "Age_15_19_yr_P": None,
    "Age_20_24_yr_P": "18-34",
    "Age_25_34_yr_P": "18-34",
    "Age_35_44_yr_P": "35-54",
    "Age_45_54_yr_P": "35-54",
    "Age_55_64_yr_P": "55+",
    "Age_65_74_yr_P": "55+",
    "Age_75_84_yr_P": "55+",
    "Age_85ov_P": "55+"
}

language_mapping = {
    # "English Only"
    # "Other Language At Home"
    "Lang_used_home_Oth_Lang_P":"Other Language at Home",
    "Lang_used_home_Eng_only_P": "English Only"

}

# G04 Age by Sex (all persons). Maps to (sex, age group); 0-17 excluded.
_g04_age_bands = {
    "18-34": ["18", "19", "20_24", "25_29", "30_34"],
    "35-54": ["35_39", "40_44", "45_49", "50_54"],
    "55+": ["55_59", "60_64", "65_69", "70_74", "75_79",
            "80_84", "85_89", "90_94", "95_99", "100_yr_over"],
}
sex_age_mapping = {
    f"Age_yr_{band}_{sex_code}": (sex, age_group)
    for age_group, bands in _g04_age_bands.items()
    for band in bands
    for sex_code, sex in [("M", "Male"), ("F", "Female")]
}

# G11 Proficiency in Spoken English by Year of Arrival by Age.
# NOTE: covers persons born overseas only, and the youngest band is 15-24
# (so 18-34 includes 15-17 year olds). Language not stated is excluded.
_g11_age_bands = {
    "18-34": ["A15_24", "A25_34"],
    "35-54": ["A35_44", "A45_54"],
    "55+": ["A55_64", "A65_74", "A75_84", "A85ov"],
}
language_age_mapping = {
    f"{band}_{lang_code}": (language, age_group)
    for age_group, bands in _g11_age_bands.items()
    for band in bands
    for lang_code, language in [("SEO_T", "English Only"),
                                ("UOLSE_T_T", "Other Language at Home")]
}

# --- Independent 18+ marginals for IPF (used by utils/census.py) ---

# G04 Age by Sex, 18+ only.
sex_mapping = {col: sex for col, (sex, _) in sex_age_mapping.items()}
age18_mapping = {
    f"Age_yr_{band}_P": age_group
    for age_group, bands in _g04_age_bands.items()
    for band in bands
}

# G49 Highest Non-School Qualification: Level of Education (persons 15+).
# 'School Education' is derived in census.py as 18+ population minus all qualifications.
education_mapping = {
    "P_PGrad_Deg_Total": "University Education",
    "P_GradDip_and_GradCert_Total": "University Education",
    "P_BachDeg_Total": "University Education",
    "P_AdvDip_and_Dip_Total": "TAFE Education",
    "P_Cert_Lev_Tot_Total": "TAFE Education",
}

# G17 Total Personal Income (Weekly). Bracket -> {demographic: weight}.
# Annual thresholds $45K and $125K are $865.38/wk and $2,403.85/wk; brackets
# that straddle them are split linearly. Income not stated is excluded.
_INCOME_LOW = 45_000 / 52
_INCOME_HIGH = 125_000 / 52
_low_split = (_INCOME_LOW - 800) / (1000 - 800)
_high_split = (_INCOME_HIGH - 2000) / (3000 - 2000)
income_brackets = {
    "Neg_Nil_income": {"<$45K": 1.0},
    "1_149": {"<$45K": 1.0},
    "150_299": {"<$45K": 1.0},
    "300_399": {"<$45K": 1.0},
    "400_499": {"<$45K": 1.0},
    "500_649": {"<$45K": 1.0},
    "650_799": {"<$45K": 1.0},
    "800_999": {"<$45K": _low_split, "$45-125K": 1 - _low_split},
    "1000_1249": {"$45-125K": 1.0},
    "1250_1499": {"$45-125K": 1.0},
    "1500_1749": {"$45-125K": 1.0},
    "1750_1999": {"$45-125K": 1.0},
    "2000_2999": {"$45-125K": _high_split, "$125K+": 1 - _high_split},
    "3000_3499": {"$125K+": 1.0},
    "3500_more": {"$125K+": 1.0},
}

# G37 Tenure and Landlord Type (occupied private dwellings).
tenure_mapping = {
    "O_OR_Total": "Own Home Outright",
    "O_MTG_Total": "Mortgage Holder",
    "R_Tot_Total": "Renter",
}
