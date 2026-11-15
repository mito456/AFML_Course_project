# Incremental predictive value audio rejector

## Method and provenance

Binary top-1 usefulness (`audio correct AND camera wrong`) misses cases where audio increases the correct-class probability without flipping top-1. This experiment therefore defines `delta = log p_fused(y) - log p_camera(y)`, with a fixed 95/5 camera/audio posterior fusion. Positive delta means lower multiclass log loss after acquiring audio. Source `clip_events.prob_source` was checked to be `oof_site_cv` for every source row, and the supplied five site folds have no site overlap. Thus delta is constructed from OOF expert posteriors, not in-sample predictions.

The rejector inputs are camera posterior/uncertainty, fixed visual pooling, IR, and placement only. It never receives audio outputs, labels/species, sites/projects, annotations, or delta. 90/10 is a pre-specified fusion sensitivity analysis, not a tuned weight.

## Targets and validation

Continuous regression predicts expected delta. Binary variants predict `delta > 0` and the pre-specified meaningful thresholds `delta > 0.002` and `delta > 0.01`; those thresholds were chosen as small and practical log-loss changes, not selected on target data. All fitting is outer site-OOF; the source selection rule is greatest OOF total true delta at a fixed pre-specified 30% budget. Results are noisy because delta is the difference of two imperfect OOF models.

| Fusion camera weight | Mean delta | Median | SD | Positive rate |
|---:|---:|---:|---:|---:|
| 0.95 | 0.02301 | -0.04706 | 0.22737 | 23.91% |
| 0.90 | 0.02138 | -0.09645 | 0.33322 | 23.91% |

## Primary source-OOF comparison (30% budget, 95/5 fusion)

| Policy | Macro-F1 | Log loss | Total delta | Mean delta/call |
|---|---:|---:|---:|---:|
| camera uncertainty | 0.5072 | 1.5588 | 29.7723 | 0.05555 |
| continuous incremental | 0.5066 | 1.5547 | 37.0570 | 0.06914 |
| binary incremental | 0.5070 | 1.5531 | 39.9772 | 0.07458 |

`feature_ablation_metrics.csv` and `fold_metrics.csv` contain every model/ablation/fold. `policy_comparison.csv` includes camera-only, audio-always, common-seed random, uncertainty, the existing binary rejector, and both incremental approaches at matching 10/20/30/50% budgets. `site_bootstrap_ci.csv` has 1,000 paired site-bootstrap intervals versus uncertainty for macro-F1, log loss, total benefit, and benefit/call. CIs condition on the fitted OOF scores and do not erase model-selection uncertainty.

## Cost and target transfer

`cost_sensitivity.csv` uses pre-specified normalized costs 0.001/0.005/0.01. Continuous acquisition requires predicted delta above cost; binary acquisition requires estimated positive-benefit probability above cost divided by a disclosed 0.10 unit-benefit scale. `target_transfer.csv` applies only the source-selected model to adaptation and test, separately; target labels did not influence features, selection, fitting, calibration, cost, or thresholds.

## Interpretation and limits

At the primary 30% source budget, both incremental point estimates improve log loss and accumulated delta over uncertainty, with the binary approach numerically stronger than the continuous approach (0.07458 versus 0.06914 mean delta/call). Neither paired site-bootstrap interval excludes no difference: binary-minus-uncertainty total benefit is -1.15 to 22.38 and continuous-minus-uncertainty is -4.49 to 20.33. Macro-F1 differences likewise cross zero. Therefore this is suggestive of a different log-loss acquisition signal, not evidence that either learned rejector genuinely beats camera uncertainty. Improvements in log loss need not improve macro-F1 because delta rewards calibrated correct-class probability even without a top-1 change. Visual/context features can retain site-correlated shortcuts despite grouped validation; expert posterior errors and a fixed fusion rule also propagate into delta. Small target populations and post-hoc comparison of many candidates further limit any domain-generalization claim.
