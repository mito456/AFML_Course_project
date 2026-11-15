# Member 2 selector handoff

Use `availability.csv` keyed by `video_name`. Do not consume its retrospective `usefulness` column in deployment. The frozen source selector output is `predicted_probability`; `split` identifies source/adapt/test.

For target adjustment, import `adapt_methods`. k=0 returns `unsupervised_target_midrank` ranking scores from selector scores only; it is not probability calibration. For k=1 or k=3, provide only `protocol_ids(adapt_draws.csv, k, seed)` and their separately acquired usefulness labels, exclude those IDs from `evaluation_ids`, and pass a documented source-only `source_prior`. Output rows always use: `video_name, selector_score, score_kind, method, k, seed`.

The interface rejects audio information, ground-truth species, site/project IDs and retrospective annotation fields. Seeds are exactly `0,1,2,3,4`.

Taiwan complements this handoff with Member 1's station-hour evidence: recorder-only/camera-only coverage, fixed versus adaptive schedules with target feedback, equal-budget comparisons, and station-bootstrap uncertainty. These results are authoritative in `outputs/taiwan/station/station_results.md`; Member 1 owns `taiwan/station_results.py` and related outputs. Member 2 must not modify them or apply SA-FARI recalibration functions to Taiwan.
