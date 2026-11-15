# Member 2 protocol alignment audit

| Component | Current implementation | Issue identified | Correction required |
|---|---|---|---|
| Nested validation | Five supplied grouped outer folds; grouped inner OOF selection | Confirmed site-disjoint source folds | Retained; outer OOF used for results |
| Posterior provenance | `video_name` join; source `oof_site_cv` | None found | Retained keyed validation |
| Visual embeddings | NPY rows were assigned from event position | Positional dependency was implicit | Now map via verified `feat_index` manifest; upstream generation unavailable |
| Policy comparison | Exact per-fold budgets; paired site bootstrap | No oracle; 1,000 resamples | Added oracle and 2,000 paired site resamples |
| Target transfer | Hardcoded posterior-uncertainty ridge | Did not reproduce source-selected configuration | Source-only grouped-OOF configuration is now selected, fit once, and frozen |
| Method naming | Binary usefulness and Shim had identical scored implementation | Presented as distinct names | Retained code names, marked operationally equivalent |

Confirmed facts: source rows are OOF site-CV; source sites do not cross supplied folds; posterior keys are unique; selector inputs exclude audio-derived fields, labels, site/project IDs, and target rows. Assumption explicitly retained: fixed 95/5 late fusion defines retrospective benefit. `feat_index` gives an explicit row-to-clip manifest and matches `clip_events` order/key set, but no upstream embedding-generation code or embedded identifiers are present, so independent provenance beyond this manifest cannot be verified. Visual-feature conclusions should retain that limitation.
