# Member 2: Binary audio-usefulness baseline

`usefulness = 1` exactly when audio is correct and camera is incorrect. The source target uses the existing site-OOF expert outcomes; no target clips contribute to fitting or OOF model selection.

## Model and validation

An unweighted L2 logistic regression is fitted on source clips only. Features are a fixed 32-D mean-pooling projection of the 512-D visual embeddings, the full camera class posterior, camera entropy and top-two margin, `is_ir`, and one-hot placement. It uses the five supplied site-grouped OOF folds; every fold was checked to have zero location-ID overlap between train and validation. Source rows in `availability.csv` contain usefulness-model OOF probabilities. Adapt/test rows use a final source-only fit.

The previous class-balanced objective shifted its probability scale toward 50% (mean OOF prediction 38.86% versus 8.23% prevalence). It was removed: the emitted value is an unweighted logistic probability initialized at the training-fold prevalence, not a class-weighted ranking score. Its calibration is assessed below using source OOF labels only; no target labels are used for calibration.

## Source site-grouped OOF metrics

- Clips: 1786; positives: 147 (8.23%)
- Logistic OOF: ROC-AUC 0.6087; average precision 0.1240; Brier 0.0803; log loss 0.3054; 10-bin ECE 0.0515; mean prediction 0.0865
- Prevalence-only OOF (0.0823 for every clip): ROC-AUC 0.5000; average precision 0.0823; Brier 0.0755; log loss 0.2844; 10-bin ECE 0.0000
- ROC-AUC uses the Mann-Whitney formulation: a tied positive/negative pair receives 0.5 credit. Average precision is also evaluated by tied score threshold, so the all-tied prevalence baseline has AP equal to prevalence.
- Missing-audio clips: 3 total (1 source); their label and probability are set to 0.

## Leakage and alignment checks

Passed: unique video names; `feat_index`/visual embedding row alignment; camera probability names aligned to events; five expected source folds; and no train/validation site overlap. The implementation never reads audio embeddings, AudioSet probabilities, audio confidence/prediction as features, species/other labels as features, annotation/oracle fields, or `location_id` as a feature.

## Limitations

The target is derived from two imperfect expert predictions and inherits their errors. Visual embeddings may encode environment correlations despite site-grouped validation. Calibration is evaluated only on source OOF predictions (and is not assumed to transfer to a new project); a probability alone does not specify a deployment audio-cost threshold. Target usefulness is retained in the CSV only for retrospective evaluation, never fitting.

## Budgeted ranking evaluation (rerun)
The following uses the regenerated source-only, site-grouped OOF scores in `availability.csv`. It evaluates 1,785 audio-available source clips (147 useful) and excludes adapt/test labels. All policies process identical fixed-size selections; random uses seeds 0-99.

| Budget | Policy | Precision | Recall | Useful captured | Audio usage | Correction rate |
|---:|---|---:|---:|---:|---:|---:|
| 10% | top_ranked | 9.50% | 11.56% | 17.00 | 10.03% | 0.95% |
| 10% | camera_uncertainty | 14.53% | 17.69% | 26.00 | 10.03% | 1.46% |
| 10% | random_mean | 8.12% +/- 1.81% | 9.89% +/- 2.20% | 14.54 +/- 3.24 | 10.03% +/- 0.00% | 0.81% +/- 0.18% |
| 20% | top_ranked | 10.64% | 25.85% | 38.00 | 20.00% | 2.13% |
| 20% | camera_uncertainty | 12.61% | 30.61% | 45.00 | 20.00% | 2.52% |
| 20% | random_mean | 8.13% +/- 1.19% | 19.74% +/- 2.90% | 29.02 +/- 4.26 | 20.00% +/- 0.00% | 1.63% +/- 0.24% |
| 30% | top_ranked | 11.01% | 40.14% | 59.00 | 30.03% | 3.31% |
| 30% | camera_uncertainty | 13.06% | 47.62% | 70.00 | 30.03% | 3.92% |
| 30% | random_mean | 8.10% +/- 0.97% | 29.54% +/- 3.55% | 43.42 +/- 5.23 | 30.03% +/- 0.00% | 2.43% +/- 0.29% |
| 50% | top_ranked | 10.86% | 65.99% | 97.00 | 50.03% | 5.43% |
| 50% | camera_uncertainty | 12.43% | 75.51% | 111.00 | 50.03% | 6.22% |
| 50% | random_mean | 8.20% +/- 0.72% | 49.84% +/- 4.36% | 73.26 +/- 6.41 | 50.03% +/- 0.00% | 4.10% +/- 0.36% |

Detailed protocol: `audio_selector_methodology.md`. Reproducible evaluator: `evaluate_audio_selector.py`. The top-ranked usefulness model exceeds random at all tested budgets, but camera uncertainty is stronger on this source-OOF cohort.
