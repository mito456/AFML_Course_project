# Member 4 Report: Selector Evaluation (v2, pending alternative audio)

**Status:** v2 pipeline frozen. Results are negative or inconclusive at the current audio quality.
**Script:** `evaluate_selectors.py` (v2, plus slice cell)
**Question:** Q3 (adapt vs select vs both), k = 0, new sites
**Tags:** [Ours] = computed by this pipeline. [Teammate] = reported by M2/M3, not checked by me. [Hypothesis] = not yet tested.

---

## 1. Summary

- The camera-first selection pipeline reproduces M3's baselines exactly and uses only pre-audio information for the selectors. [Ours]
- Always-both (macro-F1 0.4857) is slightly above camera only (0.4783), but the paired interval includes zero ([-0.014, 0.026]). [Ours]
- No selector is shown to beat camera only, always-both or random deferral. This is a negative result under the current, weak audio (16.95% accuracy). [Ours]
- M2's availability score carries a modest ranking signal (target ROC-AUC 0.747, interval [0.622, 0.875]), but it rests on only 19 useful clips out of 525. [Ours]
- The oracle sensor choice (0.538) shows about 5 macro-F1 points of room that no selector reaches. [Ours]
- Everything must be rerun when the alternative audio arrives.

---

## 2. Data and verification

| Item | Value | Tag |
|---|---|---|
| Target clips (adapt + test), k = 0 | 525 | [Ours] |
| Target sites | 66 | [Ours] |
| Species | 23 | [Ours] |
| Source clips used for fitting (out-of-fold) | `prob_source == oof_site_cv`, folds 0-4 | [Ours] |
| Clips where camera is wrong and fusion is right ("useful") | 19 of 525 | [Ours] |

Checks that pass on every run (expected values read from M3's `results_summary.csv`, tolerance 1e-4):

- Camera accuracy 0.5238, macro-F1 0.4783.
- Audio accuracy 0.1695, macro-F1 0.1544.
- Recomputed argmax predictions equal the `pred_camera` and `pred_audio` columns.
- Labels are mapped through `npz["classes"]`, with an error on any unknown label.
- The availability merge has the same length as the target set and no duplicate `video_name`.

---

## 3. Method

1. **Experts.** Frozen camera and audio probabilities from M3 (`clip_probs.npz`).
2. **Fusion.** Weighted average of log-probabilities with one scalar weight chosen by macro-F1 on source out-of-fold clips. Camera weight alpha = 0.88. [Ours]
3. **Policies.**
   - Always-visual (camera only) and always-both (fusion on every clip).
   - Confidence selector: use fusion when camera max-probability is below tau. Tau is tuned on source out-of-fold (tau = 0.64). The full tau sweep is plotted for context only.
   - Learned rejector: logistic regression on pre-audio features (camera probabilities, entropy, top-2 margin, infrared flag, placement). Target is "camera wrong and fusion right". Threshold tuned on site-grouped out-of-fold source probabilities.
   - Availability selector: defer to fusion on the clips with the highest M2 availability score, at fixed audio-use rates (quantile sweep).
   - Random deferral at matched audio use, averaged over seeds 0-4.
   - Oracle sensor choice: camera, unless camera is wrong and audio is right.
4. **Metrics.** Macro-F1 over all 23 classes (primary), accuracy, audio-used rate. The compute proxy is currently equal to the audio-used rate.
5. **Intervals.** 95% intervals from resampling sites (1000 draws, seed 42). Paired differences use the same site resamples for both policies.

---

## 4. Results (k = 0, 525 target clips)

### 4.1 Main policies

| Policy | Macro-F1 | Accuracy | Audio used | Tag |
|---|---|---|---|---|
| Camera only | 0.4783 | 0.5238 | 0% | [Ours] |
| Always-both | 0.4857 | 0.5371 | 100% | [Ours] |
| Confidence selector, source-tuned tau = 0.64 | 0.4869 | n/a | 63.6% | [Ours] |
| Learned rejector, source-tuned threshold | about 0.490 (read from plot) | n/a | about 47% | [Ours] |
| Oracle sensor choice | 0.538 | n/a | 100% | [Ours] |

Paired difference against camera only:

- Always-both: +0.007, interval [-0.014, 0.026]. [Ours]
- Confidence selector (tau 0.64): +0.009, interval [-0.010, 0.026]. [Ours]

Both intervals include zero. The rejector value is read from the plot, so take the exact number and its interval from `results_log.csv`.

### 4.2 Availability selector (M2 score)

Target ROC-AUC against "camera wrong and fusion right": 0.747, interval [0.622, 0.875], 19 positives. [Ours]

| Audio used | Macro-F1 | Diff vs camera | Diff vs always-both |
|---|---|---|---|
| 5.1% | 0.4835 | [0.001, 0.011] | [-0.020, 0.017] |
| 10.1% | 0.4863 | [-0.008, 0.019] | [-0.017, 0.019] |
| 15.0% | 0.4857 | [-0.010, 0.020] | [-0.016, 0.016] |
| 20.0% | 0.4871 | [-0.009, 0.022] | [-0.012, 0.013] |
| 30.1% | 0.4878 | [-0.006, 0.023] | [-0.010, 0.013] |
| 40.0% | 0.4877 | [-0.008, 0.024] | [-0.009, 0.012] |
| 50.1% | 0.4835 | [-0.013, 0.022] | [-0.010, 0.005] |
| 60.0% | 0.4845 | [-0.014, 0.026] | [-0.003, 0.003] |
| 80.0% | 0.4857 | [-0.014, 0.026] | [0.000, 0.000] |
| 100% | 0.4857 | [-0.014, 0.026] | [0.000, 0.000] |

Intervals are for the paired difference in macro-F1. [Ours]

### 4.3 Learned rejector

- Source positives (camera wrong, fusion right): 43 of 1786 (2.4%). [Ours]
- Site-grouped out-of-fold ROC-AUC on source: 0.7747. [Ours]

### 4.4 Audio and vocalisation (analysis only)

`ana_target_sound` is a continuous score (median 0.013, maximum 0.60). Audio accuracy by quartile of that score: 15.9%, 15.3%, 16.0% and 20.6% (about 131 clips each). Only the top quartile is higher, and the gap is within sampling noise. This is weak evidence for M3's explanation that most clips lack target vocalisation, not confirmation. The definition and units of the score are still to be confirmed with M3. This column comes from annotations and is never a selector input. [Ours, exploratory] [Teammate]

### 4.5 Condition slices (exploratory)

Camera accuracy by slice (read from M3's `results_slice.csv` chart, approximate): about 0.64 for day (colour), falling to about 0.37 for small animals. Arboreal, brief-visibility and small-animal slices are well below the overall 0.524. Selector performance by slice is computed by the slice cell and logged under `accuracy_slice` and `accuracy_diff_vs_*` metrics. These results are exploratory:

- Only 19 clips are "useful" overall, so most slices contain 0 to 5 of them.
- Slices overlap, and several were tested, so some intervals will exclude zero by chance.
- Slices with fewer than 30 clips or 5 sites are flagged and should not be read as findings.

---

## 5. What can and cannot be claimed

**Can claim**

- The pipeline is verified against M3's numbers and leak-free for selector inputs (pre-audio only, operating points tuned on source out-of-fold).
- With current audio, always-both and all selectors are within sampling noise of camera only and of each other.
- Availability ranks useful clips better than chance (AUC 0.75, wide interval, 19 positives).
- The oracle gap (about 5 macro-F1 points) is unclosed.

**Cannot claim**

- That audio improves performance, or that any selector beats camera only, always-both or random deferral.
- Check (a) of the one-pager (match always-both at lower audio use). Availability at 15-30% audio use has the same F1 as always-both, but the intervals are about +/-0.01-0.02, so the data cannot distinguish them.
- Check (b) (beat the best adapted single sensor within one condition). Adapted experts are not part of this run.
- That vocalisation explains the weak audio score (see 4.4).
- Anything about adaptation. M2 and M3 expect weak adaptation results. [Teammate]

---

## 6. Limitations

- **Few useful clips.** 19 of 525 target clips (and 43 of 1786 source clips) are positive for selection, so every selector comparison is underpowered.
- **Weak audio.** Audio accuracy is 16.95%, close to a majority-class guess. Conclusions may change with the alternative audio.
- **Rejector.** Few positives, about 30 features, no paired interval against baselines, and the threshold maximises macro-F1, so it lands near always-both and shows no cost saving.
- **Compute proxy** equals the audio-used rate, not AST time per clip times that rate.
- **Source audio** out-of-fold status and site-split folds are not independently confirmed beyond the `prob_source` and `cv_fold` columns.
- **Availability** scores come from M2's `build_availability.py`, which I have not reviewed for leakage (no `ana_*` or target-audio inputs).
- **Plan out of date.** The plan still states 399 clips, 61 sites, 55.1% camera and 30.6% audio. Current files give 525 clips, 66 sites, 52.4% camera and 16.95% audio.
- **Oracle naming.** This is a sensor-choice oracle. The plan's oracle uses adapted audio inside fusion, so they differ.
- **Log columns.** `accuracy`, `audio_used_rate` and `compute_proxy` are extra to the agreed `results_log.csv` spec.
- **Taiwan data.** It is species-specific, so audio winning there may reflect species rather than conditions. [Teammate]

---

## 7. Dependencies

- **Alternative audio (M2 and M3).** The audio source is one config line (`audio_source`). New results must be logged with a new version suffix and must not overwrite v1 or v2 rows.
- **k = 1 and 3, and Q2.** These need M3's v2 adapted predictions and M2's `adapt_methods.py`. They are evaluated on the 456 test clips only, averaged over seeds 0-4. Not yet run.

---

## 8. Next steps

1. Run the slice cell and report results as exploratory, with clips, sites and useful clips per slice.
2. Add paired differences for the confidence and rejector selectors, and against random deferral at matched audio use.
3. Log the AUCs and the number of useful clips as rows in `results_log.csv`.
4. Run k = 1 and 3 and Q2 recalibration once the adapted files are available. Report the gain from adapting, from selecting, and combined versus their sum.
5. Replace the compute proxy with AST time per clip times the audio-used rate.
6. Rerun everything with the alternative audio and log it as a new version.
7. Optional: project-ID variant labelled "known region".

---

## 9. Files

- `evaluate_selectors.py` (v2 and slice cell)
- `results/selector_pareto_curve_v2.png`
- `results_log.csv` (v2 rows: baselines, oracle, tuned tau, rejector, `conf_threshold_*`, `availability_quantile_*`, slice rows)
