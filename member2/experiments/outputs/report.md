# Member 2 nested site-grouped selector experiment

## Setup and leakage controls

E1 ranks `1-max(camera posterior)` without model fitting. E2 uses camera posterior, entropy and top-two margin. E3 adds `is_ir`, one-hot placement and a fixed 32-D mean pool of the visual embedding. E4 is conservative XGBoost with the exact E3 inputs. All joins are by `video_name`; no audio-derived field, correctness field, label, site ID, target/adapt row, or row order is a model input. Source fitting includes only source rows. Evaluation uses the common 1,785 audio-available source clips (147 useful).

## Nested validation and calibration

The five supplied site folds are outer folds and were checked for zero train/validation location overlap. Within every outer training set, GroupKFold by location creates inner OOF predictions for conservative grid selection by AP and for Platt calibration. The final outer model is fitted on the full outer training set; its validation labels are never used for tuning or calibration. E1's scale conversion uses the same inner-only procedure. Global pooled ranking uses calibrated OOF probabilities to address fold scale; fold-stratified ranking uses raw scores independently inside each outer fold. XGBoost uses shallow trees, 30/60 rounds, depth 1/2, strong L2/min-child regularisation and no early stopping (avoiding an additional held-out tuning role with only 147 positives).

## OOF ranking and calibration

| Method | Variant | ROC-AUC | AP | Brier | Log loss |
|---|---|---:|---:|---:|---:|
| camera_uncertainty | raw | 0.6881 | 0.1557 | 0.2211 | 0.6179 |
| camera_uncertainty | calibrated | 0.6829 | 0.1547 | 0.0733 | 0.2675 |
| lr_camera | raw | 0.6770 | 0.1268 | 0.0746 | 0.2710 |
| lr_camera | calibrated | 0.6408 | 0.1203 | 0.0749 | 0.2801 |
| lr_contextual | raw | 0.6437 | 0.1363 | 0.0757 | 0.2794 |
| lr_contextual | calibrated | 0.5886 | 0.1208 | 0.0751 | 0.2816 |
| xgb_contextual | raw | 0.6769 | 0.1295 | 0.0737 | 0.2696 |
| xgb_contextual | calibrated | 0.6411 | 0.1172 | 0.0752 | 0.2818 |

Site-resampled 95% confidence intervals and per-fold paired-comparison inputs are in `site_bootstrap_ci.csv` and `fold_metrics.csv`; they should be used instead of treating clips within a site as independent.

## Budget results

`budget_results.csv` holds every deterministic method result and all 100 random seeds. For each strategy/budget, random can be summarized directly across `seed=0..99`; precision, recall, useful captured and correction rate use the requested definitions. Fold-stratified selection can differ slightly in audio calls due to integer rounding per fold.

## Interpretation and limitations

Use the saved OOF tables for conclusions: complexity is not treated as a win by default. Results are limited by 147 positives, dependent expert-derived labels, and the finite set of sites. Calibration is source-OOF calibration only and is not claimed to transfer to target projects.
