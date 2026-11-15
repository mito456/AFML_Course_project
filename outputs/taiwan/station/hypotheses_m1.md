# Member 1 hypotheses (written before running the station-level analyses)

Written 2026-11-14, before `taiwan/station_results.py` was run.

- H1 (coverage): recorder-only owl-hours greatly outnumber camera-only owl-hours at the station level, at both river systems. Expected: recorder-only >= 5x camera-only overall.
- H2 (transfer, no adaptation): a fixed hour-by-month schedule learned on one river system and applied unchanged to the other will miss its target (80% of recorder owl-hours) at the new system.
- H3 (transfer, adaptive): the borrowed-prior schedule with nightly updating will land closer to the 80% target at the new system than the fixed schedule, at a similar or lower recorder budget.
- H4 (camera tip-off): removing the camera tip-off lowers owl-hours caught by a few points at the same threshold; the schedule does not depend on it for most of its catch.
- H5 (conditions): the recorder adds most (recorder-only share of owl-hours) at non-breeding stations and in the late-night hours, where perched owls are less likely to be in camera view.

Pre-set choices: target = 80% of recorder owl-hours; settings tuned only on the source river system; seeds 0-4; 95% intervals by resampling stations (2,000 resamples).
