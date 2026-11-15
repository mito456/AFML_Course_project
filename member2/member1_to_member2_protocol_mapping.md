# Member 1 → Member 2 protocol mapping

This maps the audited station-hour protocol to Member 2’s clip-level SA-FARI evaluation. Tags distinguish direct code evidence from cross-task interpretation.

| Member 2 component | Member 1 equivalent | Implementation status / comparability |
|---|---|---|
| Camera-only | Coverage tables partition camera-only, both, recorder-only, neither (`coverage()`; station_results.py:85-146). | **[Code verified]** Descriptive camera coverage, not a camera-only prediction/selection policy in transfer evaluation. Use a clearly defined camera-only clip metric separately. |
| Always-audio | Implicit reference: all eligible station-hours would be on; capture denominator is detections from this recorder schedule. Earlier adaptive docs call this 100% hours / 100% capture (adaptive_schedule.py:18-22). | **[Code verified]** Not emitted as a row in final `transfer_results.csv`; easy conceptual counterpart. |
| Equal-budget random audio | `random hours, same budget as adaptive` reports expected capture equal to adaptive recorder-on percentage (station_results.py:229-235). | **[Code verified]** Analytical expectation, not repeated random target clip draws or a random-baseline CI. For clip evaluation, state whether actual seeded random draws are used. |
| Learned selection policy | Adaptive station rule: source hourly prior plus selected target-audio outcomes, 30-day decay, camera cue and 10% audit; threshold source tuned (adaptive_schedule.py:66-94; station_results.py:164-174). | **[Code verified]** It is rule-based online adaptation, not learned clip scoring. Comparable at the policy/evaluation level only, not architecture level. |
| Fixed/source-only selection | Source month × hour detection-rate table and source-selected cutoff (station_results.py:150-161). | **[Code verified]** Strongest direct counterpart to frozen source-only Member 2 selection: no target labels/updates. |
| Frozen target evaluation | Fixed schedule is entirely frozen. Adaptive setting is frozen but state updates nightly from selected target observations. | **[Code verified]** Do not call Member 1 adaptive transfer “frozen target evaluation” without saying “threshold frozen, online target updating retained.” |
| Oracle | Retrospective target recorder-positive-hours-only selection (station_results.py:222-227). | **[Code verified]** Valid as an audio-acquisition upper bound when Member 2 has the eventual clip audio label; not deployable. |
| Detection capture rate | `sum(on × recorder_detected) / sum(recorder_detected)` (station_results.py:80-82). | **[Code verified]** Audio-label recall among always-audio positive eligible units. Align denominator to Member 2’s always-audio labelled clips. |
| Equal budgets | `% on` across eligible target station-hours; adaptive/fixed curves interpolated at shared percentages (equal_budget_ci.py:39-45). | **[Code verified]** For clips, use fraction/cost of eligible clips acquired; never compare at unmatched realized budgets. |
| Location-level paired bootstrap | Station-resampling with replacement; same weights for both curves/difference (equal_budget_ci.py:53-73). | **[Code verified]** Direct blueprint: resample locations/sites as whole clusters and calculate both policies and their difference inside every draw. |
| Source-only hyperparameter tuning | Fixed cutoff and adaptive threshold selected only from source (station_results.py:157-174). | **[Code verified]** Member 2 should state the analogous source-only tuning and hold it fixed before target scoring. |

## Critical level-of-analysis differences

| Dimension | Member 1 station-level | Member 2 clip-level implication |
|---|---|---|
| Decision unit | Turn recorder on for an eligible station-hour. | Acquire/process an audio clip. Clip duration, overlap and trigger mechanism must be reported. |
| Available target feedback | Selected-hour audio labels update later decisions nightly. | Frozen target evaluation should have no target-label feedback; any online clip adaptation is a different reported condition. |
| Camera information | A detection earlier in a night and in previous sampled night can force audio for remaining hours/night. | Camera-only score/detection can be a clip feature, but causal timing must prevent use of same/future clip outcomes. |
| Cost | Percentage of sampled recorder-on station-hours. No final energy/camera cost. | Audio acquisition/inference cost may be per clip and unequal by duration; define budget units. |
| Label / evaluation unit | Binary recorder presence in a 10-minute sampled window, aggregated at station hour. | Clip annotation/prediction label; correlation within location and recording session may differ. |
| Shift | Two river systems that also coincide with campaign/year. | Define target location/domain split explicitly; location bootstrap cannot remedy a confounded shift definition. |

## Recommended comparable reporting

1. **[Inferred]** Report an “eligible units” count and the exact denominator before every target result, matching M1’s both-sensors-available station-hour restriction.
2. **[Inferred]** Provide camera-only, always-audio, random-at-identical-budget, source-only frozen learned selection, and oracle (when eventual audio labels exist) on one target curve/table.
3. **[Inferred]** Separate “source-only frozen” from any policy that updates with acquired target audio. If offering online adaptation, label it separately and specify the feedback timing.
4. **[Inferred]** At each shared audio budget, compute the paired policy difference within each location bootstrap draw; report the 2.5/97.5 percentiles, number of locations, and draws. M1 uses 2,000 draws and RNG seed 0.
5. **[Inferred]** Describe random baseline generation. M1 reports only its expected value, so an empirical random baseline with seed variation would not be numerically identical even if conceptually aligned.
6. **[Inferred]** Report capture relative to always-audio positives and keep system-level camera+audio coverage distinct from audio-label capture; M1’s final transfer reports the latter.

## Terminology safe for shared reporting

* **[Code verified]** “Source tuned, target threshold frozen” accurately describes both M1 fixed and adaptive setting selection.
* **[Code verified]** “Online target updating” must accompany M1 adaptive results because acquired target recorder labels affect subsequent nights.
* **[Inferred]** Use “location-level paired bootstrap” for Member 2 only if each bootstrap draw resamples whole locations and evaluates all compared policies under the same draw, matching M1’s paired curve comparison.

