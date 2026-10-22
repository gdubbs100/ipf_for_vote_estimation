"""Estimate party vote and demographic shares for the 2022 Victorian districts.

Example:
    uv run python main.py --survey sep2026 --seed-matrix 2022-results

Each run writes runs/<timestamp>_<seed-matrix>_<survey>/ containing
estimates.gpkg (2022 district geometry + shares), run.log and config.json.
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.ipf import (
    PARTY,
    POPULATION_BASES,
    SEED_MATRIX_METHODS,
    SURVEYS,
    ZERO_SHARE_METHODS,
    build_demogs,
    calc_party_shares,
    fit_demog_party_margins,
    fit_full,
    get_estimated_shares,
    get_population_base,
    get_seed_matrix,
    load_survey,
    margin_errors,
)
from utils.spatial import calculate_2021_to_2022_mapping, load_2021_seds, load_2022_districts

logger = logging.getLogger("ipf")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--survey", required=True, choices=list(SURVEYS))
    parser.add_argument("--seed-matrix", default="all-ones", choices=SEED_MATRIX_METHODS)
    parser.add_argument("--zero-method", default="epsilon", choices=ZERO_SHARE_METHODS,
                        help="how 0.0 vote shares are handled in the 2022-results seed")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="floor for --zero-method epsilon")
    parser.add_argument("--population-base", default="adult-citizen", choices=POPULATION_BASES)
    parser.add_argument("--convergence-rate", type=float, default=1e-10)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser.parse_args(argv)


def setup_run_dir(args) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.runs_dir / f"{timestamp}_{args.seed_matrix}_{args.survey}"
    run_dir.mkdir(parents=True)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(run_dir / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    return run_dir


def main(argv=None):
    args = parse_args(argv)
    run_dir = setup_run_dir(args)
    logger.info("Run directory: %s", run_dir)
    logger.info("Config: %s", (run_dir / "config.json").read_text())

    districts_2022 = load_2022_districts()
    mapping = calculate_2021_to_2022_mapping(load_2021_seds(), districts_2022)

    population = get_population_base(args.population_base, mapping).sort_index()
    logger.info(
        "Population base %r: %.0f total across %d districts. Sets district size only; "
        "the demographic mix is the census all-adult mix.",
        args.population_base, population.sum(), len(population),
    )
    demogs = build_demogs(mapping, population)
    districts = population.index

    demographics, topline = load_survey(args.survey)
    logger.info("Survey %s topline: %s", args.survey, topline.to_dict())
    pct = calc_party_shares(demographics, parties=PARTY)
    topline_p = population.sum() * topline.to_numpy(dtype=float) / topline.sum()

    logger.info("Fitting demographic x party margins")
    ipf_mats = fit_demog_party_margins(demogs, pct, topline_p, args.convergence_rate)

    shape = (len(districts), *[len(d.get_columns()) for d in demogs], len(PARTY))
    seed = get_seed_matrix(args.seed_matrix, shape, list(districts), args.zero_method, eps=args.epsilon)
    logger.info("Fitting full array %s with %s seed", shape, args.seed_matrix)
    X = fit_full(seed, demogs, ipf_mats, args.convergence_rate)

    errors = margin_errors(X, demogs, ipf_mats, topline_p)
    logger.info("Max absolute margin errors:\n%s", errors.to_string())

    shares = get_estimated_shares(X, demogs, districts)
    shares.insert(0, "population", population)
    estimates = districts_2022.merge(shares, left_on="DISTRICT", right_index=True, how="left")
    out_path = run_dir / "estimates.gpkg"
    estimates.to_file(out_path, driver="GPKG")

    statewide = estimates[PARTY].mul(estimates["population"], axis=0).sum() / estimates["population"].sum()
    logger.info("Statewide estimated shares: %s", statewide.round(4).to_dict())
    logger.info("Wrote %d districts to %s", len(estimates), out_path)


if __name__ == "__main__":
    main()
