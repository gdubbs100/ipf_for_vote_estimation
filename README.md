# ipf_for_vote_estimation

Estimates first-preference vote shares by party for each of Victoria's 88 Legislative Assembly districts (2022 boundaries). It combines a statewide opinion poll with census demographics using **iterative proportional fitting (IPF)**.

A statewide poll reports how each demographic group says it will vote, for example renters or 18–34 year olds. The census reports how many people in each group live in each district. IPF finds a district × demographic × party array that matches both, so districts with more renters inherit more of the renter vote. A *seed matrix* can also carry each district's 2022 election result into the fit. That gives the model a sense of local political lean that demographics alone miss.

## How it works

The fitted array has the axes `[district, education, age, income, language, sex, tenure, party]`. Fitting runs in two stages:

1. **Demographic × party margins.** For each demographic, the poll's party shares for that group are fitted with IPF so that the rows match the census totals for each group and the columns match the poll's topline party vote. This makes every demographic table imply the same statewide result.
2. **Full array.** Starting from the seed matrix, IPF fits the full array to two sets of margins:
   - census district × demographic counts, crosswalked to 2022 districts
   - the demographic × party tables from stage 1

The district × party margin of the result gives the estimated vote share in each district. The district × demographic margins are also written out, so you can check the demographic mix the estimates rest on.

Census data is published on 2021 ABS State Electoral Divisions, which predate the 2022 redistribution. The script spreads each 2021 division's counts across the 2022 districts it overlaps, in proportion to the overlapping area ([utils/spatial.py](utils/spatial.py)).

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```bash
uv sync
```

The census tables and 2022 election results the script needs are committed under [data/](data/). The boundary shapefiles under `data/spatial/` are also needed (see [Data sources](#data-sources)).

## Running the IPF script

```bash
uv run python main.py --survey sep2026
```

Recommended settings, from the [seed comparison](claudes_analyses/seed_comparison_report.md):

```bash
uv run python main.py --survey sep2026 --seed-matrix 2022-combined --upper-house-weight 0.5 --apply-swing
```

Run `uv run python main.py --help` for the full list of options.

### Arguments

| Argument | Default | Choices / type | Description |
|---|---|---|---|
| `--survey` | *required* | `aug2026`, `sep2026` | Which poll to fit to. Selects the demographic breakdown CSV and the matching topline row in `topline_primary_vote.csv`. |
| `--seed-matrix` | `all-ones` | `all-ones`, `2022-results`, `2022-upper-house`, `2022-combined` | Starting array for the full fit. `all-ones` gives no district pattern beyond demographics. `2022-results` seeds each district with its 2022 lower house first preferences. `2022-upper-house` uses 2022 upper house first preferences counted by district. `2022-combined` mixes the two houses. |
| `--seed-weight` | `0.5` | float, 0–1 | Weight on the 2022 shares in any 2022 seed. The remainder is equal shares for each party. `1` uses the 2022 shares only. `0` is equivalent to `all-ones`. Lower values limit how much a party's 2022 geography drives its estimates. |
| `--upper-house-weight` | `0.5` | float, 0–1 | Only used with `2022-combined`. Weight on upper house shares; the remainder is lower house shares. |
| `--apply-swing` | off | flag | Shift every district's 2022 seed shares by the poll's statewide swing since the 2022 election, in percentage points. Requires a 2022 seed. |
| `--swing-floor` | `0.01` | float | Minimum party share after `--apply-swing`, before each district is renormalised. Catches districts where a party's 2022 share was smaller than its statewide loss. |
| `--zero-method` | `smoothing` | `smoothing`, `epsilon` | How to handle 0% shares in the 2022 seeds, for example parties that didn't stand in a district. IPF can never scale a zero up. `smoothing` blends each district with the statewide shares. `epsilon` floors shares at `--epsilon`. |
| `--epsilon` | `1e-6` | float | Floor used by `--zero-method epsilon`. |
| `--alpha` | mean district formal vote | float | Prior strength, in pseudo-votes, for `--zero-method smoothing`. The default gives roughly a 50/50 mix of each district's own 2022 result and the statewide result. Lower values keep more of each district's own pattern. |
| `--population-base` | `adult-citizen` | `adult`, `adult-citizen`, `formal-votes-2022` | Population each district's census counts are scaled to. This sets district size only; the demographic mix within a district stays the census adult mix. `adult` is persons aged 18+. `adult-citizen` approximates citizens aged 18+. `formal-votes-2022` is each district's 2022 formal vote. |
| `--convergence-rate` | `1e-10` | float | IPF convergence tolerance, passed to `ipfn`. |
| `--runs-dir` | `runs` | path | Parent folder for run outputs. |

### Outputs

Each run creates `runs/<timestamp>_<seed-matrix>_<survey>/` containing:

| File | Contents |
|---|---|
| `estimates.gpkg` | 2022 district boundaries with the population base, estimated party shares and estimated demographic shares for each district. Opens in QGIS or with `geopandas.read_file`. |
| `run.log` | Log of the run, including the topline, applied swing, maximum margin errors and statewide estimated shares. |
| `config.json` | The arguments used, for reproducing the run. |

[run_analysis.ipynb](run_analysis.ipynb) loads and maps run outputs.

## Data sources

| Data | Source | Location | Used for |
|---|---|---|---|
| **Poll results** | DemosAU Victoria polls, August and September 2026 ([survey_docs/](survey_docs/)) | `data/demographic_primary_vote_*.csv`, `data/topline_primary_vote.csv` | First-preference vote by demographic and topline party vote. The topline file also holds the 2022 election result, used to calculate swing. |
| **Census** | ABS 2021 Census General Community Profile (GCP), State Electoral Divisions, Victoria | `data/census/2021 Census GCP State Electroral Division for VIC/` | Counts by demographic for each 2021 division (see table below). |
| **2021 SED boundaries** | ABS Australian Statistical Geography Standard (ASGS) Edition 3, State Electoral Divisions 2021, GDA2020 | `data/spatial/SED_2021_AUST_GDA2020_SHP/` | The geography census counts are published on. |
| **2022 district boundaries** | Vicmap Admin, `STATE_ASSEMBLY_2022` layer, via DataVic (Creative Commons licence) | `data/spatial/ll_gda2020/.../VMADMIN/STATE_ASSEMBLY_2022.shp` | Target geography, and the area crosswalk from 2021 SEDs. |
| **2022 lower house results** | Victorian Electoral Commission (VEC), district results pages, scraped by [utils/vec_results.py](utils/vec_results.py) | `data/vec_2022_first_preference*.csv` | `2022-results` and `2022-combined` seeds, and the `formal-votes-2022` population base. |
| **2022 upper house results** | VEC "Results by Voting Centre" spreadsheets for each region, counted by district, scraped by [utils/vec_upper_house.py](utils/vec_upper_house.py) | `data/vec_2022_upper_house_*.csv` | `2022-upper-house` and `2022-combined` seeds. |

### Census tables

| Demographic | Groups | GCP table | Notes |
|---|---|---|---|
| Age | 18–34, 35–54, 55+ | G04 | Persons aged 18+. |
| Sex | Male, Female | G04 | Persons aged 18+. |
| Education | School, TAFE, University | G49 | TAFE = diploma or certificate; University = bachelor or higher. School = adults minus everyone with a non-school qualification. |
| Income | <$45K, $45–125K, $125K+ | G17 | Weekly bands split linearly at the annual thresholds. The 15–17 year old share of the 15–19 band is removed using single-year ages from G04. |
| Tenure | Own outright, Mortgage, Renter | G37 | Dwelling counts weighted by persons per dwelling (currently a flat 2.5 for every tenure). |
| Language | English only, Other language | G01 | All ages, rescaled to the population base. |
| Citizenship | — | G01 | Citizen rate for the `adult-citizen` population base. |

### Party groups

Results and polls use five groups: **Labor**, **L/NP** (Liberal, Nationals and joint Coalition tickets), **Greens**, **One Nation**, and **Others** (all other parties and independents).

### Notes on the 2022 results

- **Narracan.** The 2022 lower house poll failed after a candidate died. The lower house results use the January 2023 supplementary election, at which Labor did not stand. The upper house poll went ahead as normal, so the upper house results use 2022.
- **One Nation** stood in only 5 lower house seats in 2022, so the lower house seed gives a poor picture of where its vote is. This is the main reason the seed comparison recommends `2022-combined`.

To refresh the VEC results, run the scrapers from the repo root:

```bash
uv run python -m utils.vec_results
uv run python -m utils.vec_upper_house
```

## Analyses

[claudes_analyses/](claudes_analyses/) holds supporting analyses. Each script runs from the repo root, for example `uv run python claudes_analyses/seed_comparison.py`.

- **[Seed matrix comparison](claudes_analyses/seed_comparison_report.md)** ([seed_comparison.py](claudes_analyses/seed_comparison.py)). Compares the four seeds on district spread and swing against 2022, a leave-one-demographic-out test, and One Nation's district rankings. **It recommends `2022-combined` with upper house weight 0.5.** That seed is nearly as good as the lower house seed for Labor, the Coalition and the Greens. It removes most of the distortion from where One Nation stood in 2022, and it keeps the independent pattern that the upper house seed loses. `all-ones` is too flat and produces implausible swings.
- **Seed weight sensitivity** ([seed_weight_sensitivity.py](claudes_analyses/seed_weight_sensitivity.py)). Shows how `--seed-weight` drives the spread and maximum of One Nation's estimated district vote.

## Limitations

- **Census marginals are independent.** The model knows each district's age mix and income mix, for example, but not how age and income combine there. IPF fills in the joint distribution from the seed and the poll.
- **Population base sets district size only.** It doesn't adjust for differences in turnout or citizenship between demographic groups.
- **The crosswalk assumes uniform density** within each 2021 division. This is weakest where boundaries moved between rural and urban areas.
- **Estimates are compressed toward the topline.** The model takes almost all of its information about how a group votes from the poll. The seed comparison found modelled spread across districts is about half of 2022's under default smoothing.
- **Poll sampling error passes straight through.** Small demographic subsamples in the poll aren't adjusted or smoothed.

## Project layout

```
main.py                    CLI entry point for the IPF fit
utils/ipf.py               seed matrices, population bases and IPF fitting
utils/census.py            census marginals for each 2021 SED
utils/mappings.py          census column -> demographic group mappings
utils/spatial.py           boundaries and the 2021 -> 2022 crosswalk
utils/vec_results.py       VEC lower house scraper and loader
utils/vec_upper_house.py   VEC upper house scraper and loader
run_analysis.ipynb         exploring run outputs
analysis.ipynb             original exploratory notebook
claudes_analyses/          seed comparison and sensitivity analyses
data/                      polls, census, election results and boundaries
survey_docs/               source poll reports
```
