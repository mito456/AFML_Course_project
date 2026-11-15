# Taiwan Grass-Owl evidence (authoritative Member 1 results)

Source: `../../outputs/taiwan/station/station_results.md`; implementation owned by Member 1: `../../taiwan/station_results.py`. This handoff neither copies nor changes those outputs.

Taiwan's decision unit is a station-hour, where a recorder is scheduled alongside a camera. Across 13 stations and 1,666 owl-hours, the authoritative report gives 1,311 recorder-only detections (78.7%), 81 camera-only (4.9%), and 274 both. These are deployment coverage measures, not SA-FARI classification metrics; the recorder and camera have different footprints.

The report compares fixed schedules transferred between river systems with an adaptive schedule that uses a borrowed prior, nightly target feedback, and camera tip-offs. At equal recorder-on budgets it reports target-station capture and 2,000-resample station-bootstrap intervals. Differences are uncertain because only 5 or 8 target stations contribute in each transfer and every reported adaptive-minus-fixed interval includes zero. Camera tip-offs contribute materially in the reported analysis, so the adaptive policy is not a camera-independent recorder rule.

Ecological and geographical limits include two Taiwan river systems, breeding-station concentration, effort-standardized hours, possible label noise, and sensor-footprint differences. This complements SA-FARI's camera-triggered clip question—whether audio helps classification—but does not validate a shared calibration, a common acquisition score, or applying `SAFARI/adapt_methods.py` to Taiwan.
