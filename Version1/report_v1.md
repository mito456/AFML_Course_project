# Member 4 Report: <!--  -->

**Status:** Pipeline baseline only. Not a final results report.
**Script:** `evaluate_selectors.py` (v1)
**Question:** Q3 (adapt vs select vs both), k = 0, new sites
**Tags:** [M4] = computed by this script. [Hypothesis] = not yet tested. [Teammate] = reported by other members, not checked by me.

---

## 1. Summary

- Built an evaluation harness that merges `clip_events.csv` with `clip_probs.npz` on `video_name`, reproduces M3's v1 baselines exactly, and compares camera-only, always-both, a camera-confidence selector, random deferral and an oracle sensor choice. [M4]
- Always-both (0.486) is slightly above camera alone (0.478). The oracle sensor choice is 0.538. [M4]
- The confidence selector sits above random deferral over most of the curve, but no interval supports a gain over camera-only or always-both yet. [M4]
- Audio is weak at this target (16.95% accuracy), so the room for selection is limited and the pipeline, not the result, is the main deliverable at this stage.

---

## 2. Data and verification

| Item | Value | Tag |
|---|---|---|
| Target clips (adapt + test) | 525 | [M4] |
| Target sites | 66 | [M4] |
| Species (classes in npz) | 23 | [M4] |
| Test-only clips (used for k > 0) | 456 | [M4] |
| Source rows used for fusion fitting | OOF only (`prob_source == oof_site_cv`, folds 0–4) | [M4] |

Checks that pass (read from M3's `results_summary.csv`, tolerance 1e-4):

- Camera accuracy 0.5238 and macro-F1 0.4783 match `global_visual`.
- Audio accuracy 0.1695 and macro-F1 0.1544 match `audio_only`.
- Recomputed argmax predictions equal the `pred_camera` and `pred_audio` columns.
- Labels are mapped through `npz["classes"]` with an error on any unknown label.

---

## 3. Method and Execution

1. **Experts.** Frozen camera and audio probabilities from M3 (`clip_probs.npz`).
2. **Fusion.** Weighted average of log-probabilities, with one scalar weight chosen by macro-F1 on source out-of-fold clips. Fitted camera weight alpha = 0.88. [M4]
3. **Policies compared.**
   - Always-visual (camera only).
   - Always-both (fusion on every clip).
   - Confidence selector: use fusion when camera max-probability is below tau, sweeping tau from 0.05 to 1.0 (30 points).
   - Random deferral at the same audio-used rates, averaged over seeds 0–4.
   - Oracle sensor choice: camera, unless camera is wrong and audio is right (upper bound for selection).
4. **Metric.** Macro-F1 over all 23 classes (primary), accuracy, audio-used rate. The compute proxy is currently set equal to the audio-used rate.
5. **Intervals.** 95% intervals from resampling sites (1000 draws, seed 42).
6. **How to Run:**
   - Ensure Python 3.10+ and required packages (`numpy`, `pandas`, `scikit-learn`, `matplotlib`) are installed.
   - Place `clip_events.csv`, `results_summary.csv`, and `clip_probs.npz` in the working directory.
   - Execute the script:
     ```bash
     python evaluate_selectors.py
     ```
   - Outputs include terminal verification metrics, the Pareto selection plot (`results/selector_pareto_curve.png`), and updated logs in `results_log.csv`.

---

## 4. Results (k = 0, 525 target clips)

| Policy | Macro-F1 | Audio used | Tag |
|---|---|---|---|
| Camera only (always-visual) | 0.478 | 0% | [M4] |
| Always-both (fusion) | 0.486 | 100% | [M4] |
| Oracle sensor choice | 0.538 | 100% | [M4] |
| Confidence selector, best point on curve | about 0.489 | about 50–60% | [M4], tuned on test, do not claim |
| Confidence selector at about 12% audio use | about 0.487 | about 12% | [M4], untested |

Intervals are written to `results_log.csv` per row but are not reproduced here. They must be added before any number is quoted in a final report.

**What the curve shows**

- The selector is above random deferral for most activation rates, which is a first sign that camera confidence carries some information. [M4]
- The curve is jagged (for example a dip at about 22% audio use), so noise is large relative to the differences. [M4]
- Always-both beats camera by about 0.008, with no interval on the difference. [M4]
- Room for selection from camera to oracle is about 6 macro-F1 points. None of that gap is closed by a validated selector yet. [M4]

---

## 5. What can and cannot be claimed

**Can claim**

- The pipeline reproduces M3's baselines exactly and uses only pre-audio information in the selector.
- Fusion fitted on source OOF log-probabilities gives always-both at or slightly above camera-only.
- The oracle gap is about 6 macro-F1 points.

**Cannot claim yet**

- Any selector gain over camera-only or always-both. There is no paired interval, and tau is swept on test.
- Check (a) of the one-pager (match always-both at lower audio use). The point near 12% audio use looks close but is untested.
- That vocalisation explains the weak audio score. The continuous `ana_target_sound` score (median 0.013, max 0.60) shows audio accuracy of 15.9%, 15.3%, 16.0% and 20.6% across quartiles (about 131 clips each). Only the top quartile is higher, and the difference is within sampling noise, so this is weak evidence, not confirmation. The score definition and units are still to be confirmed with M3. [M4, exploratory]
- Anything about adaptation. Adapted experts are not part of this script.[Teammate]

---

## 6. Known limitations of v1

- **Tau tuned on test.** Operating point should be chosen on source OOF and reported once on test.
- **Selector is only the threshold baseline.** No learned rejector, no availability variant, no project-ID variant.
- **k = 0 only.** No k = 1 or 3, no Q2 recalibration, no conditions analysis.
- **No paired bootstrap** (selector minus random, selector minus always-both).
- **Source audio OOF status unconfirmed.** Camera rows are confirmed OOF via `prob_source`; the same should be confirmed for audio. Folds should also be confirmed as split by site.
- **Oracle naming.** This is a sensor-choice oracle. The plan's oracle uses adapted audio inside fusion, so they differ.
- **Compute proxy** equals the audio-used rate, not AST time per clip times that rate.
- **Plan out of date.** The plan still states 399 clips, 61 sites, 55.1% camera and 30.6% audio. Current files give 525 clips, 66 sites, 52.4% camera and 16.95% audio.
- **Extra log columns** (`accuracy`, `audio_used_rate`, `compute_proxy`) are not in the agreed `results_log.csv` spec.
- **Old log rows.** Rows from earlier runs with different experiment names may remain and may contain wrong numbers. Back up and remove by hand.

---

## 7. Dependency: alternative audio

Teammates are working on an alternative audio source.

- The audio source should be a single config line, so v1 and later results can be compared.
- New results must not overwrite v1 rows in the log. Use a version suffix in experiment names.
- Taiwan station data is highly species-specific, so audio winning there may reflect species rather than conditions. This is a limitation to state plainly. [Teammate]
- The proposed chimp dataset (separate audio and video) could give a cleaner test, but only if its size is adequate, labels line up and species are not confounded. [Unverified]
- Switching audio changes Q3's setup, so the group plan should be updated.

---

## 8. Next steps

1. Confirm the definition and units of `ana_target_sound` with M3. Test whether the top-quartile audio accuracy (20.6%) differs from the other quartiles using a site-resampled interval. Compare with the majority-class and project-majority baselines on the same 525 clips. This is analysis only; the column is never a selector input.
2. Add a paired site bootstrap (1000 draws): selector minus always-visual, minus always-both, minus random.
3. Choose tau on source OOF, then report once on test.
4. Train the learned rejector on source OOF using only pre-audio inputs (camera probabilities, entropy, placement, night flag), with a penalty for deferring when camera is right and fusion is wrong. Compare to the threshold baseline.
5. Add the availability variant when M2's `availability.csv` arrives.
6. Run k = 1 and 3 on the 456 test clips, then Q2 recalibration.
7. Add area under the curve, gap to the oracle and a real compute proxy.
8. Rerun everything when the alternative audio arrives and log as a new version. Do not overwrite v1 rows.

---

## 9. Files

- `evaluate_selectors_v1.py` 
- `results/selector_paretr_curve.png`
- `results_log.csv` (rows: `baseline_always_visual_*`, `baseline_always_both_*`, `oracle_sensor_choice_*`, `conf_threshold_*`)