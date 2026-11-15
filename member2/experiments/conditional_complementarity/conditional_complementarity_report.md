# Conditional multimodal complementarity

## Research question and data

Does the **existing audio classifier posterior** provide information beyond the existing camera posterior under particular conditions? This is a retrospective posterior-fusion analysis, not an acquisition policy. `clip_probs.npz` (2311 unique `video_name` keys, 23 classes) was joined to `clip_events.csv` by `video_name`; no row order was used. Label-supported source has 1786 clips from 184 sites. Source results use the supplied site-held-out predictions already produced by the current pipeline; fixed 95/5 and 50/50 arithmetic fusion was then applied without subgroup-specific tuning. Target adapt and test use the same source-derived 95/5 weight and are reported separately. Site/project IDs, species, annotations, and any audio-derived quantity are not model inputs.

## A. Camera-confidence quartiles

`confidence_analysis.csv` gives all requested metrics for camera, 95/5, and equal fusion. In the lowest-confidence quartile, 95/5 changes macro-F1 by +0.012 and log loss by -0.058; in the highest-confidence quartile it had macro-F1 0.796, accuracy 0.924, and a prediction-change rate of 0.000. Thus any apparent low-confidence top-1 gain should be treated cautiously alongside the multiple subgroup comparisons and its interval-free, descriptive nature. Quartiles are ranked into equal-sized groups to handle tied posterior values; no weights were selected within a group.

## B. Camera top-two-margin quartiles

`margin_analysis.csv` repeats the same complete comparison, including true-class probability, probability of the original camera class, changed-prediction correctness, correction, and harm rates. In the lowest-margin (most visually ambiguous) quartile, 95/5 changes macro-F1 by +0.008, while log loss changes by -0.052. A higher maximum posterior is not interpreted as better confidence unless the true-class and outcome measures agree.

## C. Environmental subgroups

`environment_analysis.csv` reports IR/non-IR and arboreal/terrestrial/unknown-placement comparisons, with paired camera-to-fusion deltas. IR is treated only as an IR/low-light proxy, not as night. These observational subgroup differences are not causal claims.

## D. Species (post-hoc)

`species_analysis.csv` is retrospective only and flags fewer than 20 clips as insufficient. Among species with at least 20 source clips, the five largest fusion log-loss decreases are: four-eyed opossum (-0.149), northern tamandua (-0.072), red-billed currasow (-0.061), night monkey (-0.050), common opossum (-0.045). This ranking is exploratory and species identity must not become an acquisition-selector input.

## E. Population/domain comparison

| Population | n | sites | Camera macro-F1 | 95/5 macro-F1 | Difference | Camera log loss | 95/5 log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_oof | 1786 | 184 | 0.507 | 0.507 | +0.000 | 1.575 | 1.552 |
| target_adapt | 69 | 30 | 0.461 | 0.461 | +0.000 | 1.693 | 1.668 |
| target_test | 456 | 64 | 0.480 | 0.480 | +0.001 | 1.641 | 1.626 |

The source headline is camera macro-F1 0.507 versus 95/5 fusion 0.507; log loss changes by -0.023. Target results are fixed-choice evaluations, not target-selected improvements. They do not justify a claim of improved domain robustness unless their paired differences and intervals are convincingly favorable.

## Statistical uncertainty and interpretation

`bootstrap_intervals.csv` contains 1,000 paired site-cluster bootstrap resamples for macro-F1, log loss, and mean true-class-probability differences, for each population, confidence/margin quartile, and environmental subgroup. Each resampled site carries all of its clips. These intervals condition on the existing modality predictions: they do **not** capture uncertainty from retraining either model. The many quartile, environment, and species comparisons are exploratory and should not be interpreted as confirmatory without multiple-comparison control and independent validation.

This can establish only whether this existing audio classifier's predictions add information beyond this existing camera classifier. It cannot show that raw audio lacks information, that a stronger audio model or representation-level fusion cannot help, or that audio can recover camera-untriggered events. Retrospective fusion also differs from an acquisition policy: both predictions are available here, whereas a policy must decide whether to process audio before its prediction exists. The next policy experiment should therefore use only permitted pre-audio signals and prospectively validate any pre-specified conditional benefit; absent robust conditional intervals, prioritizing camera adaptation is better supported.
