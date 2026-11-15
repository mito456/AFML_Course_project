# Data dictionary

## `data/processed/batrack_pairs.csv`

One row per audio-triggered recording at BatRack station BR01 (3,317 rows, 27 Jul to 5 Aug 2020). Built by `scripts/clean_batrack.py` from `audio_trigger_performance_data.csv`.

| Column | Type | Meaning |
| --- | --- | --- |
| `clip_id` | str | Audio file stem, e.g. `2020-07-27T21_02_15`. Primary key |
| `station` | str | Always `BR01` in this set |
| `rec_time_local` | datetime (Europe/Berlin) | Recording start, from the audio file name |
| `rec_time_utc` | datetime (UTC) | Same, in UTC |
| `video_path`, `audio_path` | str | Relative path inside the source tar (`matched_videos/…mp4`, `audio_trigger/…wav`) |
| `pair_gap_s` | float | Seconds between the video and audio file-name times |
| `pair_ok` | 0/1 | 1 if `pair_gap_s` ≤ 5 s (14 rows fail; one is 8 h off) |
| `video_bat` | 0/1/blank | Bat visible on video (source `y`/`n`; `e` and blanks left empty) |
| `video_bat_raw` | str | Source value, kept for audit |
| `audio_bat` | 0/1/blank | Bat calls in the audio (source `Notes`: `n` → 0, `y…` → 1) |
| `audio_uncertain` | 0/1 | 1 if the source note had `?` (`y?`) |
| `audio_code` | str | Suffix after `y` in the source note: `mb` (272), `pp` (182), `bbarb`, `p`, `pp0`. Probably species codes (*Myotis bechsteinii*, *Pipistrellus pipistrellus*, *Barbastella barbastellus*). **Unconfirmed: ask the authors** |
| `audio_note_raw` | str | Source value |
| `social_call` | 0/1/blank | Social calls present (filled for ~200 rows) |
| `behaviour` | str | Video behaviour when a bat is visible: swarming, passing, emerging, hunting |
| `n_bats` | int | Bats on video |
| `night` | date | Evening date of the night (local time − 12 h) |
| `hour_local` | int | Local hour of recording |
| `sunset_utc`, `sunrise_utc` | datetime (UTC) | Computed with astral at 50.90° N, 8.68° E (fitted to the source's own sunset times, mean error ~2.5 s) |
| `mins_since_sunset`, `mins_to_sunrise` | float | Minutes from sunset / to next sunrise |
| `is_dark` | 0/1 | Between sunset and sunrise |
| `outcome` | str | `both`, `audio_only`, `video_only`, `neither`; blank if either label missing |

## `data/processed/grassowl_hourly.csv`

One row per station-hour (13,081 rows). Built by `scripts/clean_grassowl.py`.

| Column | Type | Meaning |
| --- | --- | --- |
| `station` | str | 13 stations: GR01-03, JR, LG, YS01-02, ZW02-06, ZW08 |
| `river_system` | str | Zengwen-Yanshui (ZW, YS) or Kaoping (GR, JR, LG). **Kaoping mapping inferred** |
| `date` | date | Calendar date |
| `hour` | int | 0-6 and 17-23 only (night sampling) |
| `night` | date | Evening date (hours 0-6 belong to the previous evening) |
| `month` | int | Calendar month |
| `campaign` | str | `2023-24` (Feb 2023-Jan 2024, 8 stations) or `2024-25` (Mar 2024-Mar 2025, 5 stations) |
| `breeding` | 0/1 | Breeding station (source `breeding_state` Y/N) |
| `camera` | 0/1/blank | Perch camera detected an owl that hour (source `C-presence`) |
| `recorder` | 0/1/blank | Acoustic recorder detected an owl that hour (source `R-presence`; 10 min sampled per hour) |
| `outcome` | str | `both`, `camera_only`, `recorder_only`, `neither`; blank if either missing |

## `data/processed/grassowl_daily.csv`

One row per station-day (1,022 rows). Same columns minus `hour` and `night`, plus:

| Column | Meaning |
| --- | --- |
| `camera_from_hourly`, `recorder_from_hourly` | Max of the hourly flags for that station and date |
| `agrees_with_hourly` | 1 if the daily flags equal the hourly maximum (946 rows), 0 if not (19), blank if hourly data missing (57) |

## Auwahi hoary bats 2018 (`scripts/clean_auwahi.py`)

Times are local (`Pacific/Honolulu`). `night` is the evening date; events after midnight keep the previous evening's night.

**`auwahi_video_events.csv`**: one row per thermal-video event (384).

| Column | Meaning |
| --- | --- |
| `video_event_id` | `V001`… from the source `Event_ID` |
| `turbine` | 2, 4, 5 or 7 |
| `night`, `start_local`, `end_local`, `duration_s` | When the event happened |
| `video_file` | Source clip name (video not released) |
| `rpm` | Rotor speed during the event |
| `flight_type` | Source code `e`, `c`, `s` (306 / 55 / 23 events); meanings in the USGS report, not the data |
| `bats_visible`, `passes` | Bats in view and passes in the event |
| `chase`, `nacelle_proximity` | 0/1 |
| `hour_local`, `mins_since_sunset` | Context (sunset from approximate site coordinates) |

**`auwahi_acoustic_events.csv`**: one row per acoustic event (244), built by grouping call files on (turbine, night, `Event_ID`). 495 call files with `Event_ID` "na" are excluded.

| Column | Meaning |
| --- | --- |
| `acoustic_event_id` | `A001`…, assigned here in time order per turbine |
| `event_id_raw` | Source `Event_ID` (repeats across nights) |
| `start_local`, `end_local`, `duration_s` | First and last call file in the event |
| `n_files` | Call files in the event |
| `feeding_buzz` | 1 if any file has `Terminal` = 1 (13 files); assumed to mark terminal-phase feeding buzzes |

**`auwahi_nights.csv`**: one row per turbine-night (488). `video_hours`/`acoustic_hours` are sampling effort (blank = not sampled); `video_events`/`acoustic_events` counts; context `rpm_mean`, `precip_mm`, `wind_mean`, `wind_sd`, `rpm_starts`; `both_sampled` 0/1 (190 nights = 1).

**`auwahi_event_matches.csv`**: on `both_sampled` nights only. One row per matched pair (`both`) or unmatched event (`video_only`, `acoustic_only`). A pair matches when the events overlap after widening each by `tol_s` seconds. At 60 s: 26 pairs, 155 video-only, 175 acoustic-only.

## India Ecoacoustics metadata (`scripts/clean_ien_metadata.py`)

**`ien_files.csv`**: one row per Type A file, or per Type B file × species (11,776 rows).

| Column | Meaning |
| --- | --- |
| `media_file_name`, `archive_name`, `state` | File and the state zip it is in on Zenodo |
| `label_type` | `A` strong labels (boxes in the Zenodo annotations.csv), `B` weak file-level labels |
| `recording_date`, `latitude`, `longitude` | No time of day exists in the source |
| `species`, `taxon_rank`, `class`, `order` | From the GBIF match in the source (Type A: one species per file in metadata; all species are in annotations.csv) |
| `duration_s`, `sample_rate_khz`, `size_mb` | From the source media info |
| `habitat` | From the source notes, when given |
| `validation` | Type B only: 1 passed, 0 failed, blank not checked |
