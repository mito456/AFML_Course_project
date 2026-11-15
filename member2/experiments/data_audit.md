# Member 2 data and experiment audit (read-only)

Audit date: 2026-11-15. This document inspected existing artifacts only. No model was retrained and no existing CSV, NPZ, baseline, or result was changed.

## 1. Inventory

`member2/` contains the baseline builder (`build_availability.py`), baseline OOF availability table (`availability.csv`, 2,311 rows), the original budget evaluator and its 12-row aggregate/400-row random-seed CSVs, two SVG plots, `methodology.md`, `audio_selector_methodology.md`, hypotheses, and this instruction. `member2/experiments/` contains the nested-CV runner, its result summarizer, Python bytecode caches, and the output CSVs/reports listed below. There are no NPZ files under `member2/` itself.

The related read-only source data live in `safari/outputs/`: `clip_events.csv`, `feat_index.csv`, `split.csv`, `test_meta.csv`, three NPY embeddings/probability arrays, `clip_probs.npz`, `adapted_probs.npz.zip`, audio labels, adaptation tables, and seven upstream results CSVs. The NPY/NPZ arrays are relevant because the experiment reads visual embeddings and camera posterior arrays from them. Audio embeddings/AudioSet probabilities are inspected here but are forbidden selector inputs.

## 2. Schemas, contents, and relationships

### Core keyed tables

| File | Rows x columns | Key / relationship | Important columns and quality |
|---|---:|---|---|
| `clip_events.csv` | 2,311 x 23 | `video_name`; aligns one-to-one and in current order with `feat_index.csv` | Split, true taxon, `location_id`, audio availability, `is_ir`, placement, supplied fold, camera/audio predictions/confidences/correctness, and four annotation fields. Annotation fields are blank in 1,786 rows (source), not random missingness. No duplicate full rows. |
| `feat_index.csv` | 2,311 x 8 | `video_name`; one-to-one with events | Audio availability, RMS, duration, and frame count. No blank values or duplicate full rows. |
| `split.csv` | 2,311 x 4 | `video_name`; split/taxon/site cross-check | No blanks or duplicate full rows. |
| `test_meta.csv` | 833 x 28 | `video_name` for held-out/test metadata | Geometry, visibility and audio metadata; species/class/order/family have 3/3/7/10 blanks. It is not a source-training table. |
| `availability.csv` | 2,311 x 4 | `video_name`; generated baseline score | `usefulness`, probability and split. No blanks or duplicate full rows. Source rows are baseline OOF; other splits are final-fit scores. |

`clip_events` provides the source usefulness label: `has_audio == 1 AND correct_audio == True AND correct_camera == False`. It has 1,786 source rows; 1,785 have available audio and 147 are useful. The one unavailable source row is excluded consistently from budget evaluation. `conf_camera` should equal the maximum camera posterior and is therefore redundant with `clip_probs.npz/camera`; no row ordering should be used as a join key.

### Array files

| File | Arrays (shape, dtype, missing values) | Meaning and alignment |
|---|---|---|
| `clip_probs.npz` | `classes` (23, `<U20`); `video_name` (2,311, object); `split` (2,311, object); `camera` and `audio` (2,311 x 23, float64, 0 NaNs) | Class posterior matrices. `video_name` uniquely keys them to events; the nested runner verifies set equality. `audio` is retrospective-only. |
| `adapted_probs.npz.zip` | `classes` (23); `video_name` (525); 20 `camera_k{1,3}_s{0..4}` / `audio_k{1,3}_s{0..4}` arrays (525 x 23, float64, 0 NaNs); ten `eval_k{1,3}_s{0..4}` arrays (525, bool) | Few-shot adaptation predictions/evaluation masks. It has its own keyed 525-clip target population and must not enter source selector fitting/tuning. |
| `visual_emb.npy` | 2,311 x 512, float32, 0 NaNs | Pre-audio visual embeddings. Current experiment uses a fixed 32 x 16 mean pooling only after keyed selection. |
| `audio_emb.npy` | 2,311 x 768, float32, 0 NaNs | Audio-pipeline output; forbidden before acquisition. |
| `audioset_probs.npy` | 2,311 x 527, float32, 0 NaNs | Audio-pipeline class probabilities; labels are in `audioset_labels.csv` (527 x 2). Forbidden before acquisition. |

The numeric arrays contain no NaNs. String/object arrays need key-based, not positional, alignment. `clip_events` and `feat_index` currently have identical `video_name` sequence, but this is a validation check rather than a licence to depend on row order.

### Result and experiment CSVs

| File(s) | Dimensions and columns | Interpretation / quality |
|---|---|---|
| Original `audio_selector_budget_results.csv` | 12 x 24 | Budget, policy and aggregate precision/recall/capture/usage/correction fields; blank seed and SD/min/max fields are structural for deterministic policies. |
| Original `audio_selector_random_seeds.csv` | 400 x 8 | 4 budgets x 100 seeds; no blanks/duplicates. |
| Nested `oof_scores.csv` | 7,144 x 7 | 1,786 source rows x 4 methods: keyed score, fold, eligibility, and retrospective label. |
| Nested `ranking_metrics.csv`, `fold_metrics.csv` | 8 x 6; 20 x 8 | Overall raw/calibrated metrics and 4 methods x 5 outer folds. |
| Nested `budget_results.csv` | 832 x 9 | 32 deterministic rows plus 800 random rows. The 32 blank `seed` values are structural. |
| Nested `tuning_choices.csv`, `paired_fold_comparisons.csv`, `site_bootstrap_ci.csv` | 15 x 4; 6 x 7; 16 x 5 | Inner grids/selected settings, paired fold deltas, and 1,000 site-bootstrap intervals. |
| Upstream `results*.csv`, `shortcut*.csv`, `shift*.csv`, `adapt_draws.csv`, `adapted_results.csv` | 2–460 rows; schemas inspected | Upstream expert/shift/shortcut/adaptation summaries. They are useful context but are not selector training data. `adapted_results.csv` has structural blanks in non-applicable fields; `adapt_draws.csv` is a keyed target adaptation draw table. |

All CSVs inspected had zero duplicate *full rows*. That is not a uniqueness guarantee for every semantic key: `oof_scores.csv` deliberately repeats each `video_name` once per method, and budget tables repeat a budget for methods/seeds/strategies.

## 3. Feature eligibility and leakage risk

**Permitted before an audio acquisition:** complete camera posterior (including maximum confidence, entropy, and margin), visual embedding or a fixed visual projection, `is_ir`, placement, and audio-stream availability. Duration or other recording metadata would require a documented deployment-time availability contract before use; `n_frames` is constant (4) in the source audio-eligible cohort and has no signal here.

**Retrospective-only:** audio posterior, audio embedding, AudioSet output/labels, audio RMS (requires audio samples), audio duration if measured by decoding, audio prediction/correctness, true species/taxon, location/project, and target/adapt outcomes. Species has strong observed source heterogeneity (for example, four-eyed opossum 19/80 useful versus agouti 1/80), but it is an oracle label, not a deployable selector input.

**Prohibited oracle fields:** `correct_camera`, `correct_audio`, usefulness, any true taxon, `location_id`, annotation visibility/area/exit/target-sound fields, and target labels. Source annotation fields are blank, while test annotation fields are available; combining them would also introduce split-dependent missingness/leakage.

### Complete CSV schema register

The following compact register records every inspected CSV and its exact header. Counts above identify dimensions; blank counts are recorded where material. The upstream result CSVs are summaries (not clip-level model inputs).

| CSV | Exact columns |
|---|---|
| `safari/outputs/adapt_draws.csv` | `seed, k, video_name, species, location_id` |
| `safari/outputs/adapted_results.csv` | `sensor, k, clip_set, metric, n_clips_mean, adapted, adapted_lo, adapted_hi, global, diff, diff_lo, diff_hi, adapted_seed_min, adapted_seed_max` |
| `safari/outputs/audioset_labels.csv` | `idx, label` |
| `safari/outputs/clip_events.csv` | `video_name, split, species, location_id, has_audio, project, class, order, family, is_ir, placement, prob_source, cv_fold, pred_camera, conf_camera, correct_camera, pred_audio, conf_audio, correct_audio, ana_visible_frac, ana_mean_area, ana_exits_frame, ana_target_sound` |
| `safari/outputs/feat_index.csv` | `video_name, split, species, location_id, has_audio, audio_rms, audio_dur_s, n_frames` |
| `safari/outputs/split.csv` | `video_name, species, location_id, split` |
| `safari/outputs/test_meta.csv` | `video_id, video_name, location_id, project, placement, width, height, length_6fps, duration_s, species_list, n_species, species, n_masklets, class, order, family, visible_frac, mean_area, touches_border_frac, exits_frame, first_visible_s, last_visible_s, has_timestamp, has_audio, audio_sr, colorfulness, is_ir` |
| `safari/outputs/results/results_per_species.csv` | unnamed species index, `global_visual, audio_only, n_test, audio_minus_visual` |
| `safari/outputs/results/results_slices.csv` | `slice, n, global_visual, audio_only` |
| `safari/outputs/results/results_summary.csv` | `model, macro_f1, accuracy` |
| `safari/outputs/results/shift_drop_diff.csv` | `unit, step, metric, camera_drop, audio_drop, diff, ci_lo, ci_hi, share_audio_more` |
| `safari/outputs/results/shift_results.csv` | `sensor, level, clip_set, n_clips, n_sites, metric, value, ci_lo, ci_hi` |
| `safari/outputs/results/shortcut_night_results.csv` | `clip_set, predictor, metric, value, ci_lo, ci_hi, n_clips, n_sites, uniform` |
| `safari/outputs/results/shortcut_placement_results.csv` | `clip_set, predictor, metric, value, ci_lo, ci_hi, n_clips, n_sites, uniform` |
| `safari/outputs/results/shortcut_results.csv` | `test, clip_set, predictor, metric, value, ci_lo, ci_hi, n_clips, n_sites` |
| `member2/audio_selector_budget_results.csv` | `budget, method, seed, selected_clips, precision, recall, useful_clips_captured, audio_usage_rate, audio_correction_rate`, plus each of those five outcome fields' `_sd`, `_min`, `_max` aggregate columns |
| `member2/audio_selector_random_seeds.csv` | `budget, seed, selected_clips, precision, recall, useful_clips_captured, audio_usage_rate, audio_correction_rate` |
| `member2/availability.csv` | `video_name, usefulness, predicted_probability, split` |
| `member2/experiments/outputs/budget_results.csv` | `audio_calls, budget, correction_rate, method, precision, recall, seed, strategy, useful_captured` |
| `member2/experiments/outputs/fold_metrics.csv` | `average_precision, brier, log_loss, method, n, outer_fold, positives, roc_auc` |
| `member2/experiments/outputs/oof_scores.csv` | `calibrated_score, eligible_audio, method, outer_fold, raw_score, usefulness, video_name` |
| `member2/experiments/outputs/paired_fold_comparisons.csv` | `comparator, reference, metric, n_folds, mean_paired_difference, wins, fold_differences` |
| `member2/experiments/outputs/ranking_metrics.csv` | `average_precision, brier, log_loss, method, roc_auc, score_variant` |
| `member2/experiments/outputs/site_bootstrap_ci.csv` | `bootstrap_replicates, ci_high, ci_low, method, metric` |
| `member2/experiments/outputs/tuning_choices.csv` | `inner_trials, method, outer_fold, selected` |

## 4. Observed signals (descriptive, source audio-available cohort)

Camera uncertainty is materially associated with usefulness: usefulness rates across uncertainty quintiles were 1.4%, 2.5%, 10.4%, 14.3%, and 12.6%. Entropy has a similar pattern (1.7%, 3.4%, 9.5%, 14.3%, 12.3%); decreasing top-two margin reverses it (14.0%, 12.6%, 10.1%, 3.1%, 1.4%, from lowest to highest margin). This supports investigating uncertainty but does not establish a deployment policy.

Context rates differ: `is_ir=True` is 103/1,308 (7.9%) versus day/colour 44/477 (9.2%); placement is arboreal 39/362 (10.8%), unknown 78/866 (9.0%), terrestrial 30/557 (5.4%). These are unadjusted, site-correlated observations—not evidence that a placement policy will generalise. Source projects also range from 4.1% to 11.9%, demonstrating why project/site IDs are dangerous shortcuts.

Audio RMS and usefulness have a near-zero point-biserial correlation (-0.016) and audio duration likewise (-0.005), but both are prohibited at decision time anyway. Upstream test summaries show that camera and audio expert accuracy differ by IR, placement, visibility, size, and species; those summaries concern expert performance, not the conditional `audio correct AND camera incorrect` target and cannot be substituted for selector evidence.

## 5. Nested-CV implementation audit

**Verified:** the runner uses only `split == source`; verifies five supplied outer folds and zero source location overlap; trains scaler/model inside each outer split; uses GroupKFold by location inside outer training data; uses inner OOF AP for grid selection; keys camera/embedding data with `video_name`; excludes unavailable audio from every selector's budget pool; and deterministic `video_name` tie-breaking is not a feature. The 4-method output has exactly the same 1,785 eligible clips.

**Calibration/global ranking:** E2–E4 calibration trains a Platt model on inner OOF model predictions and outer-training labels, then applies it to the outer validation fold. This is leakage-safe. E1's Platt calibration is fitted directly on the outer-training uncertainty/labels rather than cross-fitted inner uncertainty; it still does not use outer-validation labels, but is not methodologically symmetric with E2–E4. Global pooled ranking uses calibrated OOF scores to reduce fold scale differences. Fold-stratified selection ranks raw scores within each outer held-out fold, so it does not need cross-fold scale comparability.

**Important interpretation limits:** `fold_metrics.csv`, `paired_fold_comparisons.csv`, and `site_bootstrap_ci.csv` use *calibrated* scores. Ranking conclusions should principally examine raw within-fold ranking too, because cross-fold calibration can alter global ordering and calibration quality is distinct from discrimination. The bootstrap resamples sites (appropriate cluster unit) but conditions on the already trained OOF scores; it does not repeat the full nested fitting/tuning procedure and provides no CIs for budget capture. Fold-stratified budgets use ceiling per fold (180/360/539/894 calls) versus global 179/357/536/893, so comparisons should be made within strategy, not as identical-cost comparisons across strategies. Within a given strategy, methods use identical counts.

**Current evidence:** raw OOF AP is E1 0.1557, E2 0.1268, E3 0.1363, and E4 0.1295. This does not establish E1 as a final winner: the differences are finite-source estimates, model comparison reporting is partly on calibrated ranks, and no target-domain selector evaluation exists.

## 6. Recommended next experiments (ranked)

1. **Pre-specified conditional uncertainty policy.** Nested site-CV compare uncertainty bins interacted only with `is_ir` and placement, including a fixed audio budget and site-bootstrap *budget-yield* intervals. It directly tests the observed context heterogeneity without site/species leakage.
2. **Simpler calibrated selector ablation.** Under the same nested folds, compare uncertainty alone, posterior+uncertainty, and uncertainty plus the two context variables; report raw fold ranking separately from global calibrated ranking. This resolves whether visual pooling contributes anything beyond a robust scalar.
3. **Cluster-resampled nested-CV sensitivity.** Repeat the entire tuning/evaluation pipeline over site resamples or use repeated grouped splits, reporting uncertainty for AP and budget capture. This addresses the current bootstrap's conditional-on-model limitation.
4. **Post-hoc species/order audit only.** Quantify site-aware uncertainty around taxon rates and compare with target expert results, explicitly without using taxon as a selector feature. This is scientifically informative but cannot directly yield a deployable policy.
5. **Target-domain evaluation only when outcomes are available.** Apply fixed source-trained policies to a prospectively labelled target set; no target labels may influence model/hyperparameter/calibration selection.

## Bottom line

The data support camera uncertainty as a useful source-domain signal, and placement/IR merit carefully controlled interaction tests. They do not yet support a final global policy, a species-conditioned policy, or the claim that contextual/XGBoost models are inferior under domain shift. The next experiment should be the pre-specified, low-dimensional conditional uncertainty policy with site-aware budget-yield uncertainty.
