# Member 2 — Nested site-grouped audio-usefulness experiment

## Setup, validation, and leakage prevention

The four selectors are E1 camera uncertainty (`1-max camera posterior`), E2 camera-only L2 logistic regression, E3 contextual L2 logistic regression, and E4 contextual XGBoost. E2 uses the class posterior, entropy, and top-two margin. E3/E4 add `is_ir`, one-hot placement, and a fixed 32-D mean-pooled visual projection. No audio-derived feature, correctness/outcome field, label, target/adapt row, site ID, or assumed row ordering is used as a feature. Events, embeddings, and posterior arrays are checked and joined by `video_name`.

Five supplied site-disjoint folds are the outer evaluation. Within each outer training partition, GroupKFold over sites generates inner OOF scores used for AP-based hyperparameter selection and Platt calibration. Thus outer validation labels are never used to select, train, or calibrate. Global pooling ranks cross-fitted calibrated probabilities; fold-stratified ranking ranks the raw score inside each outer fold. The common retrospective cohort contains 1,785 audio-available source clips and 147 useful clips.

E2 tunes `C` in {0.03, 0.1, 0.3, 1}. E4 is intentionally conservative: depth {1,2}, 30/60 estimators, learning rate .05, `min_child_weight=5`, `reg_lambda=10`, `reg_alpha=.2`, and 0.8 row/column subsampling. No early stopping is used, to avoid creating another label-bearing validation role with only 147 positives.

## OOF ranking and calibration

| Method | Score | ROC-AUC | AP | Brier | Log loss |
|---|---|---:|---:|---:|---:|
| camera_uncertainty | raw | 0.6881 | 0.1557 | 0.2211 | 0.6179 |
| camera_uncertainty | calibrated | 0.6829 | 0.1547 | 0.0733 | 0.2675 |
| lr_camera | raw | 0.6770 | 0.1268 | 0.0746 | 0.2710 |
| lr_camera | calibrated | 0.6408 | 0.1203 | 0.0749 | 0.2801 |
| lr_contextual | raw | 0.6437 | 0.1363 | 0.0757 | 0.2794 |
| lr_contextual | calibrated | 0.5886 | 0.1208 | 0.0751 | 0.2816 |
| xgb_contextual | raw | 0.6769 | 0.1295 | 0.0737 | 0.2696 |
| xgb_contextual | calibrated | 0.6411 | 0.1172 | 0.0752 | 0.2818 |

Raw scores are the appropriate ranking view; calibrated scores are reported separately because calibration can change rankings across folds. E1 is strongest by raw AP (0.1557); neither contextual model improves it.

## Fixed-budget results

### global_pooled_calibrated

| Budget | Method | Calls | Useful | Precision | Recall | Correction rate |
|---:|---|---:|---:|---:|---:|---:|
| 10% | camera_uncertainty | 179 | 26 | 14.53% | 17.69% | 1.46% |
| 10% | lr_camera | 179 | 25 | 13.97% | 17.01% | 1.40% |
| 10% | lr_contextual | 179 | 22 | 12.29% | 14.97% | 1.23% |
| 10% | xgb_contextual | 179 | 19 | 10.61% | 12.93% | 1.06% |
| 20% | camera_uncertainty | 357 | 45 | 12.61% | 30.61% | 2.52% |
| 20% | lr_camera | 357 | 37 | 10.36% | 25.17% | 2.07% |
| 20% | lr_contextual | 357 | 37 | 10.36% | 25.17% | 2.07% |
| 20% | xgb_contextual | 357 | 41 | 11.48% | 27.89% | 2.30% |
| 30% | camera_uncertainty | 536 | 70 | 13.06% | 47.62% | 3.92% |
| 30% | lr_camera | 536 | 67 | 12.50% | 45.58% | 3.75% |
| 30% | lr_contextual | 536 | 64 | 11.94% | 43.54% | 3.59% |
| 30% | xgb_contextual | 536 | 63 | 11.75% | 42.86% | 3.53% |
| 50% | camera_uncertainty | 893 | 112 | 12.54% | 76.19% | 6.27% |
| 50% | lr_camera | 893 | 109 | 12.21% | 74.15% | 6.11% |
| 50% | lr_contextual | 893 | 97 | 10.86% | 65.99% | 5.43% |
| 50% | xgb_contextual | 893 | 101 | 11.31% | 68.71% | 5.66% |

### fold_stratified_raw

| Budget | Method | Calls | Useful | Precision | Recall | Correction rate |
|---:|---|---:|---:|---:|---:|---:|
| 10% | camera_uncertainty | 180 | 26 | 14.44% | 17.69% | 1.46% |
| 10% | lr_camera | 180 | 22 | 12.22% | 14.97% | 1.23% |
| 10% | lr_contextual | 180 | 23 | 12.78% | 15.65% | 1.29% |
| 10% | xgb_contextual | 180 | 24 | 13.33% | 16.33% | 1.34% |
| 20% | camera_uncertainty | 360 | 45 | 12.50% | 30.61% | 2.52% |
| 20% | lr_camera | 360 | 40 | 11.11% | 27.21% | 2.24% |
| 20% | lr_contextual | 360 | 42 | 11.67% | 28.57% | 2.35% |
| 20% | xgb_contextual | 360 | 42 | 11.67% | 28.57% | 2.35% |
| 30% | camera_uncertainty | 539 | 70 | 12.99% | 47.62% | 3.92% |
| 30% | lr_camera | 539 | 71 | 13.17% | 48.30% | 3.98% |
| 30% | lr_contextual | 539 | 65 | 12.06% | 44.22% | 3.64% |
| 30% | xgb_contextual | 539 | 75 | 13.91% | 51.02% | 4.20% |
| 50% | camera_uncertainty | 894 | 111 | 12.42% | 75.51% | 6.22% |
| 50% | lr_camera | 894 | 113 | 12.64% | 76.87% | 6.33% |
| 50% | lr_contextual | 894 | 104 | 11.63% | 70.75% | 5.83% |
| 50% | xgb_contextual | 894 | 110 | 12.30% | 74.83% | 6.16% |

### Random reference (100 seeds)

| Strategy | Budget | Useful captured, mean +/- SD | Precision, mean +/- SD |
|---|---:|---:|---:|
| global_pooled_calibrated | 10% | 14.54 +/- 3.24 | 8.12% +/- 1.81% |
| global_pooled_calibrated | 20% | 29.02 +/- 4.26 | 8.13% +/- 1.19% |
| global_pooled_calibrated | 30% | 43.42 +/- 5.23 | 8.10% +/- 0.97% |
| global_pooled_calibrated | 50% | 73.26 +/- 6.41 | 8.20% +/- 0.72% |
| fold_stratified_raw | 10% | 14.73 +/- 3.37 | 8.18% +/- 1.87% |
| fold_stratified_raw | 20% | 29.67 +/- 4.42 | 8.24% +/- 1.23% |
| fold_stratified_raw | 30% | 44.35 +/- 5.31 | 8.23% +/- 0.99% |
| fold_stratified_raw | 50% | 73.54 +/- 5.77 | 8.23% +/- 0.65% |

Random selections use the same eligible clips and fixed seeds in `budget_results.csv`.

## Paired folds and site-aware uncertainty

| Comparator vs E1 | Metric | Mean paired fold difference | Folds won |
|---|---|---:|---:|
| lr_camera | roc_auc | -0.0063 | 1/5 |
| lr_camera | average_precision | -0.0398 | 2/5 |
| lr_contextual | roc_auc | -0.0348 | 0/5 |
| lr_contextual | average_precision | -0.0243 | 2/5 |
| xgb_contextual | roc_auc | -0.0087 | 1/5 |
| xgb_contextual | average_precision | -0.0245 | 2/5 |

`site_bootstrap_ci.csv` gives 1,000 site-resampled 95% CIs; `fold_metrics.csv` and `paired_fold_comparisons.csv` preserve all paired inputs. The small number of positives, expert-derived target, and finite site set make close differences inconclusive. The evidence here does not support selecting a more complex contextual model over camera uncertainty. Source-OOF calibration is not assumed to transport under domain shift.
