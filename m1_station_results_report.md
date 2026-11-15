# Member 1 Report: Station-Level Results

## Executive summary

This analysis evaluates co-located acoustic recorders and cameras for Taiwan Grass-Owls across 13 stations on the Kaoping and Zengwen–Yanshui river systems. Of 12,274 station-hours in which both sensors were working, owls were detected in 1,666 hours. The recorder detected owls without a corresponding camera detection in 78.7% of those owl-hours, while the camera alone contributed 4.9%. Thus, recorder-only detections were 16.2 times as frequent as camera-only detections (95% station-bootstrap interval: 7.7–25.6).

The strongest conclusion is that, in this deployment, acoustic recording supplies substantially more owl detections than the camera. Evidence for an adaptive recorder schedule transferring better than a fixed schedule is suggestive but not statistically established: all equal-budget difference intervals include zero. The real-time camera tip-off is important to adaptive performance, contributing 11–18 percentage points at a common threshold.

## Data and analysis design

- **Study population:** Taiwan Grass-Owl hourly detections from 13 stations on two river systems.
- **Comparable effort:** 12,274 station-hours where both sensors worked.
- **Detection unit:** an *owl-hour* is a station-hour with at least one owl detection by either sensor (1,666 owl-hours).
- **Uncertainty:** 95% intervals from 2,000 station-level resamples. There are only 13 stations (Kaoping: 5; Zengwen–Yanshui: 8), so intervals are appropriately wide.
- **Scheduling outcome:** the share of recorder-detected owl-hours occurring while the recorder was scheduled on.

## 1. Sensor coverage

| Scope | Recorder only | Camera only | Both | Recorder-only / camera-only |
| --- | ---: | ---: | ---: | ---: |
| All stations (1,666 owl-hours) | 1,311; **78.7%** [70.3, 90.8] | 81; **4.9%** [3.1, 10.8] | 274; 16.4% [3.4, 25.8] | **16.2×** [7.7, 25.6] |
| Zengwen–Yanshui (941) | 652; 69.3% [63.0, 87.8] | 58; 6.2% [3.0, 30.2] | 231; 24.6% | 11.2× [2.1, 22.3] |
| Kaoping (725) | 659; 90.9% [88.4, 96.1] | 23; 3.2% [2.1, 3.8] | 43; 5.9% | 28.7× [23.9, 45.3] |

Twelve of the 13 stations had more recorder-only than camera-only owl-hours. The exception, YS02, had two camera-only owl-hours and no recorder-only detections.

## 2. External check: chimpanzee monitoring

An external co-located dataset from Issa Valley, Tanzania was analysed at the site-day level. At the main 500 m pairing radius, 800 eligible site-days contained 365 chimpanzee days. Acoustic-only detections occurred on 214 days (58.6% [47.7, 71.1]) and camera-only detections on 49 days (13.4% [8.8, 17.8]).

This reproduces the direction of the owl result, but it should not be interpreted as a direct comparison of sensor quality. The acoustic system covered an approximately 500 m audible range during half of each daytime hour; motion-triggered cameras ran continuously but had a range of at most 29 m. Results also changed materially with the definition of co-location: at 1 km, acoustic-only and camera-only shares were 32.3% and 26.8%, respectively.

## 3. Schedule transfer between river systems

Policies were tuned to catch 80% of recorder owl-hours in one river system, then frozen and evaluated in the other.

| Target system (source for tuning) | Policy | Recorder on | Owl-hours caught |
| --- | --- | ---: | ---: |
| Kaoping (Zengwen–Yanshui) | Fixed schedule | 59.1% [56.1, 64.6] | **81.9%** [77.0, 89.1] |
|  | Adaptive with camera tip-off | 32.3% [15.1, 49.5] | **56.3%** [23.3, 65.4] |
| Zengwen–Yanshui (Kaoping) | Fixed schedule | 38.8% [36.7, 40.6] | **53.8%** [49.1, 79.6] |
|  | Adaptive with camera tip-off | 35.3% [22.6, 53.5] | **88.9%** [41.2, 97.0] |

The 80% catch target did not transfer consistently. The fixed setting exceeded it at Kaoping but fell short at Zengwen–Yanshui; the adaptive setting did the reverse. This is consistent with schedule miscalibration under distribution shift, likely amplified by owl-hours being concentrated at a small number of breeding stations.

At matched recorder budgets, adaptive scheduling was essentially level with the fixed schedule at Kaoping (−1.9 to +2.8 percentage points across 30–60% budgets) and ahead by 19.3–43.2 points at Zengwen–Yanshui. However, every paired station-bootstrap interval for the difference included zero. The apparent adaptive advantage is therefore promising, not established.

Both scheduling approaches exceeded random scheduling at equal budget. The adaptive policy depended materially on its camera tip-off: removing that tip-off reduced its catch by 11–18 points at the same threshold.

## 4. Conditions associated with recorder contribution

| Condition | Recorder-only owl-hours per 100 station-hours | Recorder-only share of owl-hours |
| --- | ---: | ---: |
| Breeding stations (3) | **32.1** [22.9, 48.0] | 76.0% [66.2, 91.1] |
| Non-breeding stations (10) | **4.0** [2.0, 6.3] | 86.4% [73.4, 94.0] |
| Wet season (May–Oct) | 13.8 [6.7, 21.3] | 73.2% [54.0, 92.7] |
| Dry season (Nov–Apr) | 7.8 [1.6, 18.6] | 89.5% [77.9, 92.7] |

The recorder added more detection volume at breeding stations and during the wet season, where owls were more prevalent. Recorder-only shares were high across conditions, but there was no supported late-night effect and the breeding/non-breeding share intervals overlap.

## Conclusions against the prespecified hypotheses

| Hypothesis | Conclusion |
| --- | --- |
| Recorder-only detections are at least five times camera-only detections | **Supported** (16.2× [7.7, 25.6]) |
| A fixed schedule learned on one system misses 80% on the other | **Partly supported** |
| Adaptive scheduling is closer to 80% at a similar or lower budget | **Partly supported, not established** |
| Camera tip-off adds only a few points | **Rejected**; it adds 11–18 points |
| Recorder benefit is greatest at non-breeding stations and late at night | **Not supported** |

## Interpretation and limitations

The camera is the detection bottleneck in this particular owl deployment: the recorder captured owl-hours that the camera missed far more often than the reverse. Some of that difference reflects the sensors' unequal footprints rather than an intrinsic difference in sensing quality.

The main scheduling finding is more cautious. Timing information may transfer as a ranking signal, especially at Zengwen–Yanshui, but the 5–8 stations per river system provide insufficient precision to establish an adaptive advantage. Moreover, the adaptive policy's benefit partly assumes that camera detections can be recognised and communicated in real time, which was not verified.

Other limitations are the effort-standardized sampling design (seven nights per month; 17:00–06:00; 10 recorder minutes per hour), possible noise in unpublished owl-call labels, unmeasured camera cost, and the confounding of river system with study year. More river systems or stations are needed before treating the transfer result as a general deployment recommendation.

## Recommendation

Use acoustic recording as a core component of monitoring where the objective is to detect Taiwan Grass-Owl presence in comparable conditions. Treat adaptive scheduling as a candidate efficiency strategy, but retain a fixed schedule or conservative spot-checking until it is validated across more stations and without relying on unverified real-time camera tip-offs.

## Source materials

This report synthesizes the analysis in `station_results.md` and the accompanying `m1_station_results.zip` result package. The reported intervals are station-resampled, 95% intervals with 2,000 resamples.
