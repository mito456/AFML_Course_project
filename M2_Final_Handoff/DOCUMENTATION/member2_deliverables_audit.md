# Member 2 deliverables audit

## Dataset mapping

| Component | SA-FARI | Taiwan |
|---|---|---|
| Decision unit | Camera-triggered clip | Station-hour |
| Additional modality | Audio | Acoustic recorder |
| Main question | Is audio useful for this clip? | Is recording worthwhile at this station-hour? |
| Acquisition cost | Audio-processing rate | Recorder-on rate |
| Main evaluation | Classification benefit | Detection capture |
| Biological context | Species traits | Station/species ecological conditions |
| Adaptation | Selector recalibration | Station-policy adaptation |

## SA-FARI deliverables

Audit performed without rerunning experiments or overwriting results.

### Already completed and validated

- `hypotheses.md` exists and retains Q1, Q2, Q3 and condition hypotheses.
- `availability.csv` has 2,311 unique `video_name` values, exactly matching the 2,311 `clip_events.csv` values (1,786 source, 69 adapt, 456 test). Its probabilities are finite and within `[0,1]`.
- `build_availability.py` resolves the project root relative to its own file, uses `safari/outputs`, joins camera arrays by `video_name`, validates embedding/event alignment, and checks five source folds for zero `location_id` overlap. Source output is held-out-fold score; adapt/test output comes from the final source-only fit. Missing audio is consistently emitted with label and score zero.
- Existing methodology documents that target labels are not used for fitting/tuning and gives the fields Member 4 can consume: `video_name`, `predicted_probability`, `split` (with `usefulness` retrospective-only).

### Newly implemented

- `adapt_methods.py`: guarded k=0 target midrank score adjustment and k=1/k=3 label-supported prevalence/odds recalibration. The module only accepts precomputed selector scores plus an explicit two-column permitted-label table; forbidden audio, identity, annotation, and target-label feature fields are rejected.
- `species_traits.csv`: one row for each of the 23 observed SA-FARI classes, with cited reference URLs and explicit `unknown` / reasonable-assumption statuses.
- `test_adapt_methods.py`: lightweight unit coverage for malformed input, bounds, duplicate IDs, leakage fields, overlap, k values, schema and repeatability.

### Partial / blocked

- There is no Member 4 directory or selector implementation in this checkout. Therefore no Member-4 function signature can be inspected, and no direct import integration is possible. The stable handoff schema is documented below.
- `adapt_draws.csv` supplies the expert-adaptation IDs for each seed/k. `protocol_ids()` validates those IDs; callers must pass the usefulness labels obtained only for those clips. The draw table includes target species and location solely to define the externally approved examples; neither can enter selector features.
- No few-label selector outcome is generated: the repository provides adaptation class labels/draws, but does not supply protocol-authorized usefulness labels separate from evaluation. Creating them would require re-evaluating audio/camera outcomes and is outside the no-experiment instruction.

### Handoff interface for Member 4

Input scores: iterable of `{video_name, predicted_probability}` only. For k=0 call `no_label_rank_adjust(scores)`. For k=1/3 call `few_label_recalibrate(scores, labels, k=..., seed=..., evaluation_ids=..., expected_adaptation_ids=protocol_ids(draws_path, k, seed), source_prior=...)`. Labels are exactly `{video_name, usefulness}`; `evaluation_ids` excludes adaptation IDs; `source_prior` is explicitly source-derived. Both return rows with exactly `video_name, selector_score, score_kind, method, k, seed`.

## Taiwan project-level evidence and Member 2 interpretation

Taiwan is supporting Member 1 evidence, not a Member 2 implementation. The authoritative `outputs/taiwan/station/station_results.md` reports recorder-only versus camera-only detections, unequal sensor footprints, fixed versus target-feedback adaptive recorder schedules, equal-budget comparisons, and 2,000-resample station-bootstrap intervals. Its geographical/ecological scope is 13 stations on two Taiwan river systems, with breeding-station concentration and deployment-footprint limits. Member 2 interprets this as complementary station-level evidence only; it does not alter the policy, bootstrap outputs, or claims.
