# Member 2 final validation report

## Experimental protocol

The decision unit is a camera-triggered clip with audio available. Policies process audio for an exact per-outer-fold `ceil(budget × eligible clips)` at 10%, 20%, 30%, or 50%; the common source cohort has 1785 clips from 184 sites. Source selector fitting/configuration uses source data only with five supplied site-disjoint outer folds and grouped inner folds. Target adaptation and test are frozen evaluations: their labels are read only to calculate metrics.

## Validation audit

Camera/audio posteriors are key-joined by `video_name`; source posterior provenance is `oof_site_cv`, and supplied outer folds have no site overlap. Selector inputs are camera posterior/uncertainty, visual embedding, IR, and placement; no audio-derived input, label, site/project ID, or target metadata enters a selector. `visual_emb.npy` is mapped through the `feat_index.csv` manifest after verifying cardinality, uniqueness, key-set equality, and identical manifest/event order. This establishes the repository's declared alignment, but the checkout contains no embedding-generation script or row identifiers inside the NPY; independent reconstruction from upstream generation remains unavailable.

## Equal-budget policy comparison

Incremental benefit is the change in true-class log probability from fixed 95/5 camera/audio fusion versus camera-only. All policies use the same per-fold exact-budget convention; consequently even Oracle can include a negative-benefit clip when fewer than the budget count have positive benefit. Oracle ranks realized retrospective benefit and is an upper bound, not deployable.

| Policy | 10% benefit | 20% benefit | 30% benefit | 50% benefit |
|---|---:|---:|---:|---:|
| camera_only | 0.000 | 0.000 | 0.000 | 0.000 |
| audio_always | 41.144 | 41.144 | 41.144 | 41.144 |
| random | 2.969 | 6.480 | 11.471 | 20.579 |
| camera_uncertainty | 12.833 | 19.899 | 32.693 | 49.874 |
| binary_usefulness | 12.498 | 20.401 | 25.692 | 44.161 |
| continuous_incremental | 12.755 | 28.935 | 38.226 | 44.939 |
| binary_incremental | 8.893 | 21.392 | 25.590 | 39.782 |
| oracle | 92.002 | 100.331 | 99.450 | 86.136 |

Camera uncertainty is the simple reference. Binary usefulness predicts `audio correct AND camera wrong`; continuous incremental value predicts fusion-objective improvement. `shim_cost_sensitive` is operationally identical to binary usefulness in this run (same target, candidates, selection criterion and ranked decisions), so it is not an independent method. Random is empirical fixed-seed random acquisition, always audio is maximum acquisition, and camera-only is zero acquisition.

## Statistical comparison

`results/paired_site_bootstrap.csv` contains paired whole-site bootstrap comparisons for all four budgets: continuous vs uncertainty; binary incremental vs uncertainty; continuous vs binary incremental; and each learned policy vs random and camera-only. It uses 2,000 resamples, seed 20261115, identical resampled sites within each pair, 95% percentile intervals, and records valid sample counts. These intervals condition on fixed nested OOF predictions; they do not include uncertainty from repeated fitting, split selection, or policy/model selection. Overlapping intervals are not evidence of equivalence.

## Target transfer

`results/source_selected_transfer_policy.csv` records a continuous configuration selected with source-only inner grouped OOF data and fit once on all source clips. `results/target_transfer.csv` applies that frozen policy separately to adaptation and test at 30%; it reports clip count, acquisition rate, macro-F1, log loss, and incremental benefit. It is a source-selected frozen-policy evaluation, not target adaptation: no target feedback, thresholding, refitting, or selection occurred.

## Limitations and interpretation

Audio experts are imperfect and results depend on fixed late fusion. Domain shift and small target adaptation data limit transfer claims. Camera-triggered sampling cannot establish audio's value for events the camera never triggers. The embedding alignment has manifest-level but not independently reconstructible provenance. Bootstrap intervals understate full pipeline uncertainty, and recording versus processing budgets are different cost proxies.

Continuous incremental value has the best 30% source point estimate but earlier paired intervals included harm versus camera uncertainty, so it is promising rather than established. Oracle headroom measures retrospective opportunity under this fusion outcome, not attainable deployment performance. Target transfer is descriptive and not convincing evidence of generalization. The defensible claim is that frozen, pre-audio clip-level ranking can prioritize retrospective fusion benefit on this source protocol; it does not establish broadly adaptive audio acquisition.
