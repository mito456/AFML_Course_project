# AFML wildlife sensing: data and station-level results (Member 1)
## Repo map

```
README.md                  this file
data_manifest.csv          every dataset: source URL, DOI, licence, what is in the repo, where its media lives
requirements.txt           Python packages
.gitignore                 keeps audio, video and archives out of git
docs/data_dictionary.md    every column of every processed file

data/raw/<dataset>/        original files, unchanged (sizes and SHA-256 in data/raw/checksums.csv)
data/processed/            cleaned tables built by scripts/ (do not edit by hand)
data/media/                local media cache, git-ignored

scripts/                   one cleaning script per dataset, plus the chimpanzee analysis
taiwan/                    Taiwan Grass-Owl analysis code (run as python -m taiwan.<name>)
outputs/taiwan/station/    Member 1 deliverables: station_results.md and everything it cites
outputs/taiwan/            earlier Grass-Owl exploration (see status column below)
outputs/chimp/             chimpanzee co-location tables
```

## Data

### Raw files (`data/raw/`)

| Folder | Files | Size | Licence | Source |
| --- | --- | --- | --- | --- |
| `grassowl_taiwan/` | `Data_Hour_combine.csv`, `Data_Day_combine.csv`, `Readme.txt` | 0.3 MB | CC BY 4.0 | Mendeley doi:10.17632/tdnpwgv6s5.1 |
| `chimp_issa2018/` | `PresenceAbsence_matrices_CT_and_PAM.xlsx` | 0.2 MB | CC0 | Dryad doi:10.5061/dryad.5dv41ns34 |
| `batrack2020/` | 14 CSVs from `csv.tar` | 2.3 MB | CC BY 4.0 | doi:10.17192/fdr/67 |
| `auwahi_bats2018/` | `video_detections.csv`, `acoustic_detections.csv`, `nightly_summary.csv` | 0.2 MB | CC0 | USGS doi:10.5066/P937H9LQ |
| `ien_india2026/` | `type_A_metadata.csv`, `type_B_metadata.csv` | 11 MB | CC BY-NC | Zenodo 18743214 / 18927866 |
| (top level) | `birdclef2024_species.csv` | 8 KB | not verified | species list from a public solution repo |
| (top level) | `checksums.csv` | 4 KB | n/a | sizes and SHA-256 of the files above |

### Processed files (`data/processed/`)

| File | Built by | Rows | Used for |
| --- | --- | --- | --- |
| `grassowl_hourly.csv`, `grassowl_daily.csv` | `scripts/clean_grassowl.py` | 13,081 station-hours | All Taiwan analysis |
| `batrack_pairs.csv` | `scripts/clean_batrack.py` | 3,317 clips | Clip-level video + audio pairs |
| `auwahi_video_events.csv`, `auwahi_acoustic_events.csv`, `auwahi_nights.csv`, `auwahi_event_matches.csv` | `scripts/clean_auwahi.py` | 384 video + 244 acoustic events, 488 turbine-nights | Event-level complementarity |
| `ien_files.csv`, `birdclef2024_ien_overlap.csv` | `scripts/clean_ien_metadata.py` | 11,776 file-species rows | Choosing which India archives to download |

Not in the repo: audio, video and tar archives (BatRack media alone is over 30 GB) and the BirdCLEF competition data. `data_manifest.csv` says where to download each.

## Code

### `scripts/`

| File | What it does |
| --- | --- |
| `clean_grassowl.py` | Raw Grass-Owl CSVs → `grassowl_hourly.csv`, `grassowl_daily.csv` |
| `clean_batrack.py` | BatRack CSVs → `batrack_pairs.csv` |
| `clean_auwahi.py` | Auwahi CSVs → event and night tables (`--tol SECONDS` sets the video–acoustic matching window, default 60 s) |
| `clean_ien_metadata.py` | IEN metadata → `ien_files.csv`, BirdCLEF overlap |
| `chimp_coverage.py` | Chimpanzee co-location table (M1 Task 2) → `outputs/chimp/` |

### `taiwan/`

| File | What it does | Status |
| --- | --- | --- |
| `config.py` | Every judgement call (what counts as a detection, edge hours, wet-season months, power figures) | Shared |
| `load.py` | Loads the tidy station-hour table | Shared |
| `adaptive_schedule.py` | The adaptive recorder scheduler (per-station weighted average, camera tip-off, 10% spot checks) | **Current method** |
| `station_results.py` | M1 Tasks 1, 3, 4: coverage counts, transfer between river systems, conditions (~10 min) | **M1 deliverable** |
| `equal_budget_ci.py` | 95% ranges for the equal-budget comparison (~2 min) | **M1 deliverable** |
| `outcomes.py` | Both / camera-only / recorder-only / neither, by hour and month | Earlier exploration |
| `oracle.py` | Fewest recorder-hours that keep X% of detections | Earlier exploration |
| `energy.py` | Recorder-hours → AudioMoth energy; quote energy saved, not battery days | Earlier exploration |
| `context_selector.py` | One schedule for all stations, includes the logistic regression the team dropped | Superseded |

## Outputs

### `outputs/taiwan/station/`: Member 1 deliverables

| Type | Files |
| --- | --- |
| Write-up | `station_results.md` (results, hypotheses, interpretation, assumptions), `hypotheses_m1.md` (written before running) |
| Task 1: Taiwan coverage | `coverage_overall.csv`, `coverage_by_station.csv` |
| Task 2: chimpanzee table | `chimp_coverage.csv`, `chimp_site_days.csv`, `chimp_pairs.csv`, `chimp_coverage.py` (copies of `outputs/chimp/` and the script) |
| Task 3: policy transfer | `transfer_results.csv`, `transfer_curves.csv`, `transfer.png`, `equal_budget_ci.csv`, `equal_budget_ci.py` |
| Task 4: conditions | `conditions.csv`, `conditions.png` |
| Team log | `results_log_rows.csv` (rows for the shared `results_log.csv`) |
| Run record | `run.log` (console output of `station_results`) |

### `outputs/taiwan/`: earlier exploration

| Files | From | Status |
| --- | --- | --- |
| `adaptive_station_month.csv/.png`, `adaptive_summary.csv`, `adaptive_tradeoff.csv/.png` | `adaptive_schedule` | Pooled results (not the frozen-transfer protocol; use `station/` for that) |
| `outcomes_overall.csv`, `outcomes_by_hour.csv/.png`, `outcomes_by_month.csv/.png` | `outcomes` | Exploration |
| `oracle_recorder.csv`, `oracle_any.csv` | `oracle` | Exploration |
| `selector_*.csv/.png` (8 files) | `context_selector` | Superseded |

### `outputs/chimp/`

`chimp_coverage.csv` (counts and shares at 250 m / 500 m / 1 km), `chimp_site_days.csv` (one row per acoustic site-day), `chimp_pairs.csv` (distance from every camera to every acoustic site).

## Rebuild everything

From the repo root:

```bash
pip install -r requirements.txt

# data
python scripts/clean_grassowl.py
python scripts/clean_batrack.py
python scripts/clean_auwahi.py
python scripts/clean_ien_metadata.py

# Member 1 results
python -m taiwan.station_results      # Tasks 1, 3, 4
python -m taiwan.equal_budget_ci      # equal-budget ranges
python scripts/chimp_coverage.py      # Task 2

# earlier exploration (optional)
python -m taiwan.outcomes
python -m taiwan.oracle
python -m taiwan.adaptive_schedule
```

Seeds are 0–4; intervals are 95% from 2,000 resamples of stations (or acoustic sites for the chimpanzee table).

## Known limitations

**Grass-Owl**: presence/absence per hour only, no media; effort-standardized subset (7 nights a month, 17:00–06:00, recorder 10 min per hour); labelling method unpublished; river system and year coincide; GR/JR/LG river-system grouping inferred.

**Chimpanzees**: camera data is daily, acoustic hourly, so the unit is site-day; recorders hear ~500 m but record half of each hour 06:00–19:30, cameras see ~29 m but run 24 h; results depend on the pairing radius.

**BatRack**: every clip exists because the audio trigger fired, so silent bats are missing; 3,128 of 3,317 clips are from one night; one station; `audio_code` meanings inferred.

**Auwahi**: detections only; microphone at the nacelle (footprint differs); both sensors ran on only 190 of 488 turbine-nights, so compare on `both_sampled == 1`.

**India Ecoacoustics**: date but no time of day; Type B is 79% one Madhya Pradesh survey; CC BY-NC, keep use non-commercial.

## Citation and licences

Cite the sources when using these files. Original files are unchanged in `data/raw/`; processed files rename and recode columns and add time and context features (see `docs/data_dictionary.md`).

- Chang, C.-H., Hong, S.-Y., & Tsai, J.-S. (2026). *Paired acoustic and camera trap detections of Australasian Grass-Owl (Tyto longimembris) across diel and seasonal scales in southwestern Taiwan* [Dataset]. Mendeley Data. https://doi.org/10.17632/tdnpwgv6s5.1
- Crunchant, A.-S., et al. (2020). Listening and watching: Do camera traps or acoustic sensors more efficiently detect wild chimpanzees in an open habitat? *Methods in Ecology and Evolution*. Data: Dryad. https://doi.org/10.5061/dryad.5dv41ns34
- Gottwald, J., & Lampe, P. (2021). *Data_BatRack_2020* [Dataset]. Philipps-Universität Marburg. https://doi.org/10.17192/fdr/67
- Gottwald, J., et al. (2021). BatRack: An open-source multi-sensor device for wildlife research. *Methods in Ecology and Evolution*, 12(10), 1867–1874. https://doi.org/10.1111/2041-210X.13672
- Gorresen, P.M. (2020). *Maui Island, Hawaiian hoary bat behavior at wind turbines, 2018* [Dataset]. U.S. Geological Survey. https://doi.org/10.5066/P937H9LQ
- Ramesh, V., et al. (2026). A large-scale crowd-sourced annotated acoustic dataset of Indian fauna. *bioRxiv*. https://doi.org/10.64898/2026.07.20.739496