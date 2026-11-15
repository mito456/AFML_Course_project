# Station-level results (Member 1)

Data: Taiwan Grass-Owl hourly detections, 13 stations on 2 river systems, 12,274 station-hours where both sensors were working ([doi:10.17632/tdnpwgv6s5.1](https://doi.org/10.17632/tdnpwgv6s5.1)). An "owl-hour" is a station-hour where at least one sensor detected an owl (1,666). The recorder heard owls in 1,585 of them.
Intervals: 95%, from 2,000 resamples of stations. The real unit is the station, and there are only 13 (5 at Kaoping, 8 at Zengwen–Yanshui), so intervals are wide.
Code: `taiwan/station_results.py`. Hypotheses written before running: `hypotheses_m1.md`. Status: Tasks 1–5 done; chimpanzee code: `chimp_coverage.py`.

## 1. Coverage: what each sensor catches alone

| Scope | Recorder only | Camera only | Both | Recorder-only ÷ camera-only |
| --- | --- | --- | --- | --- |
| All 13 stations (1,666 owl-hours) | 1,311 · **78.7% [70.3, 90.8]** | 81 · **4.9% [3.1, 10.8]** | 274 · 16.4% [3.4, 25.8] | **16.2× [7.7, 25.6]** |
| Zengwen–Yanshui, 8 stations (941) | 652 · 69.3% [63.0, 87.8] | 58 · 6.2% [3.0, 30.2] | 231 · 24.6% | 11.2× [2.1, 22.3] |
| Kaoping, 5 stations (725) | 659 · 90.9% [88.4, 96.1] | 23 · 3.2% [2.1, 3.8] | 43 · 5.9% | 28.7× [23.9, 45.3] |

[Ours] Per-station counts: `coverage_by_station.csv`. 12 of 13 stations have more recorder-only than camera-only owl-hours; the exception is YS02 (2 camera-only owl-hours, none heard).

## 2. External co-located table: chimpanzees, Issa Valley, Tanzania

Data: Crunchant et al. 2020, Dryad doi:10.5061/dryad.5dv41ns34 (CC0). [Source]
Setup: 12 acoustic sites (3 clusters × 4 positions, rotated every 2 weeks), recording 30 min of each hour 06:00–19:30, calls audible to ~500 m; 53 motion-triggered video cameras (21 systematic, 32 targeted), running 24 h, range ≤29 m. Both labelled by hand (spectrogram + listening; video review). [Source]

- **Time unit:** site-day (the camera table is daily, so the hourly acoustic table was collapsed to "heard on any half-hour that day"). [Ours]
- **Co-location:** a camera counts for an acoustic site if it is within 500 m of any of that site's 4 sensors (the paper's acoustic detection range). All 12 sites have 2–8 such cameras; nearest 56–270 m. [Ours]
- **Kept:** days where the acoustic site recorded and ≥1 co-located camera was working: 800 site-days, 365 with chimpanzees. [Ours]

| Pairing radius | Site-days | Chimp days | Both | Acoustic only | Camera only | Acoustic-only share | Camera-only share |
|---|---|---|---|---|---|---|---|
| 250 m | 583 | 260 | 57 | 181 | 22 | 69.6% [57.5, 83.3] | 8.5% [4.0, 13.6] |
| **500 m (main)** | **800** | **365** | **102** | **214** | **49** | **58.6% [47.7, 71.1]** | **13.4% [8.8, 17.8]** |
| 1 km | 819 | 436 | 178 | 141 | 117 | 32.3% [26.1, 40.0] | 26.8% [21.1, 32.7] |

95% intervals: 2,000 resamples of the 12 acoustic sites. [Ours]

By camera type at 500 m: systematic cameras (8 sites, 162 chimp days) acoustic-only 93.8%, camera-only 3.7%; targeted cameras (placed on trails/fruit trees; 12 sites, 352 chimp days) acoustic-only 59.7%, camera-only 12.2%. [Ours]

**Reading:** same direction as Taiwan: the recorder alone catches chimpanzees on ~4× more days than the camera alone (214 vs 49) [Ours]. But the split depends on how "co-located" is defined: widening the pairing radius to 1 km makes the two nearly even, because more cameras are counted against the same recorder. Unlike Taiwan, the camera runs 24 h while the recorder covers half of each daytime hour, so the camera has the time advantage and the recorder the area advantage (500 m vs 29 m). [Ours]

## 3. Policy transfer between river systems (Q2, station level)

Settings were tuned on the source river system to catch 80% of recorder owl-hours there, then frozen and used at the other system. "Caught" is the share of the target system's recorder owl-hours that fell in hours the recorder was on.

**At the frozen setting**

| Used at (tuned on) | Policy | Recorder on (% of station-hours) | Owl-hours caught |
| --- | --- | --- | --- |
| Kaoping (Zengwen–Yanshui), 702 owl-hours | Fixed schedule (no adaptation) | 59.1 [56.1, 64.6] | **81.9 [77.0, 89.1]** |
| | Adaptive: borrowed prior + nightly updating (threshold 0.25) | 32.3 [15.1, 49.5] | **56.3 [23.3, 65.4]** |
| | Adaptive, no camera tip-off (threshold 0.05) | 71.0 [57.1, 83.0] | 92.0 [79.9, 95.4] |
| | Random hours, same budget as adaptive (expected) | 32.3 | 32.3 |
| | Oracle | 16.0 | 100 |
| Zengwen–Yanshui (Kaoping), 883 owl-hours | Fixed schedule (no adaptation) | 38.8 [36.7, 40.6] | **53.8 [49.1, 79.6]** |
| | Adaptive: borrowed prior + nightly updating (threshold 0.13) | 35.3 [22.6, 53.5] | **88.9 [41.2, 97.0]** |
| | Adaptive, no camera tip-off (threshold 0.07) | 56.8 [47.5, 68.3] | 94.3 [66.5, 97.9] |
| | Random hours, same budget as adaptive (expected) | 35.3 | 35.3 |
| | Oracle | 11.2 | 100 |

**At equal recorder budgets** (owl-hours caught at the target system, % [95% interval]; adaptive minus fixed in points)

| Recorder on | Kaoping: fixed | Kaoping: adaptive | Kaoping: difference | Zengwen–Yanshui: fixed | Zengwen–Yanshui: adaptive | Zengwen–Yanshui: difference |
| --- | --- | --- | --- | --- | --- | --- |
| 30% | 53.1 [48.7, 56.9] | 51.2 [44.1, 63.1] | −1.9 [−9.5, +11.6] | 44.0 [37.9, 74.7] | 87.3 [42.9, 93.2] | **+43.2 [−12.8, +53.0]** |
| 40% | 65.3 [59.3, 70.0] | 68.1 [51.9, 72.6] | +2.8 [−12.8, +11.1] | 55.4 [49.2, 80.7] | 90.4 [57.6, 96.1] | **+35.0 [−10.1, +45.6]** |
| 60% | 83.2 [79.1, 88.2] | 85.8 [75.8, 88.8] | +2.6 [−8.6, +7.3] | 76.8 [72.0, 92.0] | 96.1 [78.7, 98.3] | **+19.3 [−4.7, +25.7]** |

Random hours catch the same % as the budget (30, 40, 60). [Ours] Intervals: 2,000 resamples of target stations; each resample rebuilds both curves and reads them at the budget (`equal_budget_ci.py`, `equal_budget_ci.csv`).

- Every difference interval includes zero. At Zengwen–Yanshui the adaptive schedule was ahead in about 70% of resamples; at Kaoping about 55%. With 8 and 5 stations, and owl-hours concentrated in a few breeding stations, the gain is large in the pooled data but not established. [Ours]
- The adaptive schedule cannot run below about 23–29% recorder-on, because of the 10% spot checks and camera tip-offs. At 30% some resamples did not reach the budget (30% at Kaoping, 20% at Zengwen–Yanshui); those are left out of the interval. A 20% row shown in the earlier version was the curve's lowest point (25%/23% on), not a true 20% value, and has been removed. [Ours]
- Both always beat random hours at the same budget. [Ours] Curves: `transfer.png`, `transfer_curves.csv`.

**Camera tip-off dependence** [Ours]: at the same threshold (0.10) on the source systems, removing the tip-off drops owl-hours caught from 88.3% to 70.0% (Zengwen–Yanshui) and from 85.2% to 73.8% (Kaoping).

## 4. Conditions: where the recorder adds most

| Condition | Recorder-only owl-hours per 100 station-hours | Recorder-only share of owl-hours |
| --- | --- | --- |
| Breeding stations (3) | **32.1 [22.9, 48.0]** | 76.0% [66.2, 91.1] |
| Non-breeding stations (10) | **4.0 [2.0, 6.3]** | 86.4% [73.4, 94.0] |
| Wet season (May–Oct) | 13.8 [6.7, 21.3] | 73.2% [54.0, 92.7] |
| Dry season (Nov–Apr) | 7.8 [1.6, 18.6] | 89.5% [77.9, 92.7] |
| 17:00 and 06:00 | 0.9 and 1.5 | 86% and 100% (7 and 4 owl-hours) |
| 18:00–05:00 (each hour) | 7.7 to 16.6 | 73% to 87% |

[Ours] Full table: `conditions.csv`; chart: `conditions.png`. Season (month) stands in for the season condition that SA-FARI cannot test.

## Hypotheses (written before running)

| Hypothesis | Outcome |
| --- | --- |
| H1: recorder-only owl-hours ≥ 5× camera-only | **Supported**: 16.2× [7.7, 25.6] |
| H2: a fixed schedule learned on one system misses 80% at the other | **Partly**: missed at Zengwen–Yanshui (53.8%), met at Kaoping (81.9%, at 59% recorder-on) |
| H3: the adaptive schedule lands closer to 80% than the fixed one, at a similar or lower budget | **Partly / not established**: ahead at equal budgets at Zengwen–Yanshui (+19 to +43 points) and level at Kaoping (−2 to +3), but every difference interval includes zero; its frozen threshold overshot at Zengwen–Yanshui (88.9%) and undershot at Kaoping (56.3%) |
| H4: the camera tip-off adds only a few points | **Rejected**: it adds 11–18 points at the same threshold |
| H5: the recorder adds most at non-breeding stations and late at night | **Not supported**: no late-night effect; the recorder-only *share* is higher at non-breeding stations but intervals overlap, and in volume it adds 8× more at breeding stations |

## Interpretation

1. For this species the camera is the bottleneck: 79% of owl-hours were heard but never seen, and the camera alone added 5% [Ours]. An external co-located dataset (chimpanzees, Tanzania) points the same way, with the recorder alone on 59% of chimp days vs 13% for the camera alone, but that split shrinks to 32% vs 27% when cameras up to 1 km away are paired [Ours].
2. What the schedule learns about timing may transfer as a ranking: at equal budgets it is level with a fixed schedule at Kaoping and ahead by 19–43 points at Zengwen–Yanshui, but with 5–8 stations none of these differences is statistically clear (all intervals include zero) [Ours].
3. The cutoff does not transfer: tuned on one system it overshot or undershot the 80% target at the other (89% and 56%), because each system's owl-hours are dominated by one to three breeding stations. This is the station-level version of selector miscalibration under shift (clip-level Q2) [Ours; link to Q2 is our interpretation].
4. 11–18 points of the schedule's catch depend on the camera tip-off, which assumes owls are recognised on camera in real time; that is unverified [Ours].
5. The recorder adds most where owls are (breeding stations, wet season) in volume, while in share terms almost every hour and station type is recorder-dominated [Ours].

## Assumptions and limits

- The camera is treated as always on and cheap (motion-triggered); camera cost was not measured. [Unverified]
- The two sensors cover different areas (perch-mounted camera vs a recorder that hears beyond the perch), so part of "recorder only" is footprint, not sensor quality. [Source: dataset description]
- How owl calls were identified is unpublished; labels are treated as reference labels, possibly noisy. [Unverified]
- Effort-standardized subset only: 7 nights per month, 17:00–06:00, recorder 10 minutes per hour. [Source: dataset readme]
- River system and year coincide (Zengwen–Yanshui 2023–24, Kaoping 2024–25), accepted by the team. Kaoping grouping of GR/JR/LG inferred. [Hypothesis]
- Scheduler settings other than the threshold (starting guess worth 3 nights, 30-day half-life, 10% spot checks) were set by hand, not tuned. Seeds 0–4.
- Breeding status is a site property, so it is used here only to describe conditions, never as a selector input.
- Chimpanzee table: sensor footprints differ by ~17× in radius and recording windows differ (acoustic 06:00–19:30, half of each hour; cameras 24 h), so shares describe this deployment, not sensor quality. Acoustic rotation means each site has ~2-week blocks of data. [Source/Ours]

## Still to do

- Nothing from the Member 1 brief. A clearer equal-budget answer needs more stations (e.g. a third river system or SA-FARI sites).
