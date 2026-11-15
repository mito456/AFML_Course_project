# Open questions for Member 1

These questions remain after auditing the current `member1_akshata` code and shipped results. They are not requests to change the method.

1. **Camera label timing.** `run_station()` uses a camera positive after it is present in the offline hourly row, which cues later hours but not the same hour (adaptive_schedule.py:84-92). What real-time camera recognition latency is assumed, and is the next decision truly available within the same hour?
2. **Camera reliability.** The code has only observed-perfect versus all-zero tip-offs. Is an intended definition available for 75%/50% reliability (false negatives only; per event/hour/night; random seed)?
3. **Prior interpretation.** Should `START_NIGHTS=3` be presented as an elicited pseudo-count/empirical-Bayes prior strength, or only as a hand-set simulation parameter? The code does not validate it.
4. **Target update protocol.** Is “previous sampled night” intentional when stations have gaps, rather than previous calendar night? Code carries the flag across the next group regardless of elapsed days (adaptive_schedule.py:73, 83, 93).
5. **Adaptation variants.** Was a zero-target-update, update-first-night-only, or update-first-three-nights-only version created outside the audited branch? None exists here; `START_NIGHTS=3` does not implement first-three-night labelling/freezing.
6. **Random baseline.** Is the intended final comparison the analytical expectation used in `transfer_results.csv`, or should it be actual random schedules with their own Monte Carlo uncertainty? Current code only reports expected capture (station_results.py:229-235).
7. **Cost endpoint.** The final station transfer reports recorder-on percentage/hours but calls no energy or operating-cost function. Should shared reporting call that a recording budget/cost proxy, rather than operating cost?
8. **Result-version mismatch.** `equal_budget_ci.csv` retains 20% rows, while `station_results.md` says those were removed from the presentation because the adaptive curve cannot reliably reach that budget. Which artifact is the authoritative public table?
9. **Historical source data semantics.** In target adaptive simulation, selected target recorder labels are revealed and used immediately on later nights. Is this meant to represent actual operational audio review turnaround, and if so what turnaround delay is assumed?
10. **River-domain definition.** Kaoping station grouping is explicitly inferred in cleaning code (clean_grassowl.py:19-22), and river and campaign/year coincide. Has this grouping / the intended domain-shift interpretation been externally confirmed?
11. **Source result discrepancy.** Final report text states adaptive tuning is leave-one-source-station-out; code supports this for tuning (station_results.py:164-174), while target prior is all source data (lines 191-195). Is that deliberate and should it be made explicit in shared methods wording?
12. **Perfect-label assumptions.** What detection/annotation error is known for camera and recorder labels? The code treats available labels as binary truth and only drops sensor-missing rows (load.py:21-25).

## Explicitly absent in audited implementation

* **[Code verified]** No switch or configuration value freezes adaptive updates after 0, 1, or 3 target nights.
* **[Code verified]** No 50%/75% camera tip-off reliability model.
* **[Code verified]** No learned classifier/model-training artifact in the final station transfer pipeline.
* **[Code verified]** No final-policy camera-only or always-audio rows in `transfer_results.csv` (only coverage description and implicit references).
* **[Unknown]** Whether any of these exist in a private/uncommitted analysis; they cannot be established from the fetched branch.

