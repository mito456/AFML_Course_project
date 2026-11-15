# Taiwan Member 1 final report

**Documentation identifier:** `TAIWAN_MEMBER1_RESULTS_RECONCILIATION_2026-11-15` (documentation reconciliation, not an experiment).  
**Authoritative result set:** `member1_audit_source/outputs/taiwan/station/`; artifacts inspected 2026-11-15 10:23 UTC.  
**Audited implementation revision:** `origin/member1_akshata` at `1b11252` (per `member2/member1_implementation_audit.md`).

## Scope and protocol

This report concerns Taiwan Grass-Owl station data only. The decision unit is an eligible station-hour. Camera is treated as always available; the action is to turn the acoustic recorder on. Cost is percentage of eligible station-hours (recorder-on time), and the primary outcome is detection capture rate: recorder-detected owl-hours caught while on. This is not the SA-FARI clip-processing objective or its incremental log-probability metric.

The fixed policy is a source-derived month-by-hour schedule with a source-selected cutoff. The adaptive policy has a source-selected threshold but is **not** a frozen source-only selector: it uses selected target recorder outcomes to update later nightly decisions, with a decayed hourly-rate rule, camera tip-off, and spot checks. Random is an analytical expected-capture baseline at the adaptive policy's realized recorder-on rate. Oracle retrospectively selects target recorder-positive hours. Bootstrap intervals resample whole stations (2,000 resamples).

## Authoritative results

`coverage_overall.csv` contains 12,274 comparable station-hours at 13 stations and 1,666 owl-hours. Recorder-only detections account for 1,311 owl-hours (78.69%, 95% station-bootstrap interval 70.33--90.84), camera-only for 81 (4.86%, 3.06--10.82), and their ratio is 16.19 (7.68--25.56).

`transfer_results.csv` is the authoritative frozen-setting transfer table. With source Zengwen-Yanshui and target Kaoping, fixed captures 81.91% at 59.12% recorder on; target-updating adaptive captures 56.32% at 32.29% on. With source Kaoping and target Zengwen-Yanshui, fixed captures 53.79% at 38.83% on; target-updating adaptive captures 88.88% at 35.32% on. The adaptive results retain target feedback and must be interpreted accordingly. Oracle is 100% capture at 15.95% and 11.22% recorder-on respectively; it is retrospective only.

At equal recorder budgets in `equal_budget_ci.csv`, all adaptive-minus-fixed station-bootstrap intervals include zero: Kaoping differences at 30/40/60% are -1.87 [-9.49, 11.57], +2.80 [-12.83, 11.15], and +2.63 [-8.63, 7.30] percentage points; Zengwen-Yanshui differences are +43.24 [-12.76, 53.04], +35.05 [-10.11, 45.59], and +19.29 [-4.71, 25.71]. The 20% rows remain in the raw CSV but are not authoritative presentation points because the adaptive curve cannot reliably reach that budget; `station_results.md` explicitly removes them.

## Conclusion and limitations

The recorder contributed substantially more owl detections than camera in this deployment. Adaptive scheduling is suggestive, not established against the fixed schedule at equal budgets. Its camera tip-off and selected target-recorder outcomes are operational assumptions that matter to its performance; it cannot be described as a fully frozen target evaluation.

Key limitations are only 13 stations, owl-hour concentration at breeding stations, unequal sensor footprints, camera cost not measured, potentially noisy/unpublished call labels, and river system confounded with campaign/year. The audited code and artifacts do not establish any additional latest/private result set. `taiwan/station_results.py`, `taiwan/adaptive_schedule.py`, and `outputs/taiwan/station/equal_budget_ci.py` are the relevant implementation/analysis files; none was modified or run in this reconciliation.
