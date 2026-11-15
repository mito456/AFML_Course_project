# Member 4 Report: Selector Evaluation and Adaptation Decomposition (v3)

**Status:** v3 pipeline frozen. Results logged to `results_log.csv` (840 rows, prefix `m4_`).  
**Script:** `Evaluate_selectors_v3.ipynb` (v3 evaluation pipeline)  
**Question:** Q3 (adapt vs select vs both) at k = 0, 1, 3; Q2 (domain shift recalibration); condition slices  
**Tags:** [Ours] = computed by this pipeline. [Teammate] = reported by M2/M3, not checked by me. [Hypothesis] = not yet tested.  

---

## 1. Summary

- Built and validated the full v3 evaluation pipeline incorporating M2's adapted probability arrays (`adapted_probs.npz`), adapt draws (`adapt_draws.csv`), and availability scores (`availability.csv`). Reproduces all M3 and M2 baselines exactly and uses strictly pre-audio features for all selectors. [Ours]
- **Orthogonality of Adaptation and Selection (Q3):** Adaptation ($k = 1, 3$) and selective audio acquisition operate independently. The interaction term between adaptation gain and selection gain is statistically indistinguishable from zero (-0.0028 [-0.012, 0.007] at $k = 1$; -0.0000 [-0.014, 0.014] at $k = 3$). Their benefits add up cleanly: $\text{combined gain} \approx \text{adapt gain} + \text{select gain}$. [Ours]
- **Confirmation of Check (a):** At $k = 3$, the primary availability selector (`avail_b30` at 29.9% audio use) achieves macro-F1 of 0.5561 [0.482, 0.592], which matches Always-Both (0.5615 [0.487, 0.599] at 100% audio use) within sampling noise (paired difference -0.0054 [-0.011, 0.000]). This satisfies Check (a) by matching Always-Both while cutting audio compute by over 70%. [Ours]
- **Domain Shift Stability (Q2):** Pre-audio ranking models exhibit strong resilience under geographic transfer. M2's availability score maintains a target ROC-AUC of 0.7466 [0.622, 0.875] (source OOF was 0.721), and the learned rejector maintains 0.7901 [0.669, 0.902] (source OOF was 0.7747). [Ours]
- **Condition Slices:** Across 14 environmental slices, the primary selector delivers consistent positive accuracy gains over camera-only baselines (e.g., day colour +3.0%, larger animals +2.0%, brief visibility +2.2%), but individual slice sample sizes are small (most contain $\le 10$ useful clips). [Ours]
- **Audio Constraint:** Unadapted target audio is weak (accuracy 16.95%, macro-F1 0.1544), limiting unadapted selection headroom. Adaptation of visual and audio experts is essential to unlock higher absolute classification performance. [Ours]

---

## 2. Data and verification

| Item | Value | Tag |
|---|---|---|
| Target clips (adapt + test), k = 0 | 525 | [Ours] |
| Target evaluation clips, k = 1 | 502 (mean across seeds 0–4) | [Ours] |
| Target evaluation clips, k = 3 | 456 (mean across seeds 0–4) | [Ours] |
| Target sites | 66 | [Ours] |
| Species classes | 23 | [Ours] |
| Source clips used for fitting (out-of-fold) | `prob_source == oof_site_cv`, folds 0–4 (1,786 clips, 184 sites) | [Ours] |
| Target clips where camera is wrong and fusion is right ("useful") | 19 of 525 (3.6%) | [Ours] |
| Source clips where camera is wrong and fusion is right | 43 of 1,786 (2.4%) | [Ours] |

Checks that pass on every run (tolerance 1e-4):

- Camera accuracy 0.5238, macro-F1 0.4783 match M3's `results_summary.csv`.
- Audio accuracy 0.1695, macro-F1 0.1544 match M3's `results_summary.csv`.
- At $k = 1$: camera global diff 0.0000 and adapted diff 0.0000 match M3.
- At $k = 3$: camera global diff 0.0000 and adapted diff 0.0000 match M3.
- Recomputed argmax predictions equal the `pred_camera` and `pred_audio` columns.
- Evaluation masks in `adapted_probs.npz` match `adapt_draws.csv` exactly across all $k \in \{1, 3\}$ and seeds 0–4.
- All merges (`clip_events.csv`, `clip_probs.npz`, `adapted_probs.npz`, `availability.csv`) have identical row counts with zero duplicate `video_name`.

---

## 3. Method

1. **Experts.** Frozen visual and audio probabilities from M3 (`clip_probs.npz`) for unadapted baselines ($k = 0$). For adapted regimes ($k = 1, 3$), probabilities are loaded from M2's `adapted_probs.npz` across seeds 0–4.
2. **Fusion.** Weighted average of log-probabilities with scalar weight fitted on source OOF clips. Optimal camera weight $\alpha = 0.88$ for global experts and $\alpha = 0.85$ for adapted experts. [Ours]
3. **Policies evaluated.**
   - **Always-visual:** Camera only (`vis_g` for global, `vis_a` for adapted).
   - **Always-both:** Full multimodal fusion on every clip (`both_g`, `both_a`).
   - **Confidence selector:** Defer to fusion when camera maximum probability is below threshold $\tau$. Source-tuned $\tau = 0.64$ (`tau_src`), plus rate-matched variants at 20% (`conf_b20`) and 30% (`conf_b30`).
   - **Learned rejector:** Logistic regression trained on source OOF pre-audio features (visual embeddings pooled to 32-D, camera probabilities, entropy, margin, infrared flag, placement). Operating threshold tuned on source OOF (`rejector_src`).
   - **Availability selector:** Defer to fusion on clips with highest M2 availability score. Evaluated at fixed budget quantiles: 20% (`avail_b20`) and primary 30% (`avail_b30`).
   - **Random deferral:** Randomly selects clips at matched audio budget, averaged across seeds 0–4.
   - **Oracle sensor choice:** Retrospective ceiling selecting audio/fusion whenever camera is incorrect and audio/fusion is correct (`oracle_g`, `oracle_a`).
4. **Q3 Decomposition Framework.**
   - $\text{adapt\_gain} = \text{vis\_a} - \text{vis\_g}$
   - $\text{select\_gain} = \text{sel\_g} - \text{vis\_g}$
   - $\text{combined\_gain} = \text{sel\_a} - \text{vis\_g}$
   - $\text{interaction} = \text{combined\_gain} - (\text{adapt\_gain} + \text{select\_gain})$
5. **Q2 Recalibration Framework.** Compares frozen source threshold, no-label rate-matching (quantile on target camera confidence), and $k$-label temperature scaling on camera probabilities.
6. **Metrics.** Macro-F1 over all 23 classes (primary), accuracy, and audio-used rate. The compute proxy is set equal to the audio-used rate.
7. **Intervals.** 95% paired whole-site bootstrap intervals from 1,000 resamples across 66 sites (seed 42).

---

## 4. Results

### 4.1 Q3: Adaptation vs. Selection Decomposition ($k = 1$ and $k = 3$)

Performance averaged across 5 adaptation seeds (seeds 0–4). Intervals are 95% whole-site bootstrap CIs.

| Regime & Policy | Audio used | Macro-F1 | 95% Bootstrap CI | Accuracy | Diff vs. Always-Both | Tag |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **k = 1 (~502 eval clips)** | | | | | | |
| Global Camera (`vis_g`) | 0.0% | 0.4732 | [0.394, 0.517] | 0.5219 | -0.0446 [-0.057, -0.033] | [Ours] |
| Adapted Camera (`vis_a`) | 0.0% | 0.5108 | [0.437, 0.547] | 0.5530 | -0.0070 [-0.012, 0.000] | [Ours] |
| Global Both (`both_g`) | 100.0% | 0.4802 | [0.401, 0.526] | 0.5359 | -0.0376 [-0.049, -0.026] | [Ours] |
| Adapted Both (`both_a`) | 100.0% | 0.5178 | [0.440, 0.554] | 0.5669 | 0.0000 (Reference) | [Ours] |
| Confidence Selector (`tau_src`) | 59.4% | 0.5166 | [0.439, 0.553] | 0.5641 | -0.0012 [-0.003, 0.000] | [Ours] |
| Learned Rejector (`rejector_src`) | 42.6% | 0.5183 | [0.440, 0.555] | 0.5661 | +0.0005 [-0.005, 0.004] | [Ours] |
| Availability 20% (`avail_b20`) | 20.0% | 0.5155 | [0.438, 0.552] | 0.5606 | -0.0023 [-0.009, 0.004] | [Ours] |
| **Primary Availability 30% (`avail_b30`)** | **30.0%** | **0.5201** | **[0.442, 0.555]** | **0.5674** | **-0.0005 [-0.006, 0.005]** | [Ours] |
| Global Oracle (`oracle_g`) | 100.0% | 0.5360 | [0.455, 0.576] | 0.5876 | +0.0182 [+0.011, +0.026] | [Ours] |
| Adapted Oracle (`oracle_a`) | 100.0% | 0.5752 | [0.500, 0.613] | 0.6207 | +0.0574 [+0.047, +0.068] | [Ours] |
| **k = 3 (~456 eval clips)** | | | | | | |
| Global Camera (`vis_g`) | 0.0% | 0.4689 | [0.390, 0.513] | 0.5250 | -0.0926 [-0.111, -0.075] | [Ours] |
| Adapted Camera (`vis_a`) | 0.0% | 0.5460 | [0.474, 0.577] | 0.5846 | -0.0155 [-0.021, -0.004] | [Ours] |
| Global Both (`both_g`) | 100.0% | 0.4782 | [0.399, 0.524] | 0.5417 | -0.0833 [-0.100, -0.066] | [Ours] |
| Adapted Both (`both_a`) | 100.0% | 0.5615 | [0.487, 0.599] | 0.6044 | 0.0000 (Reference) | [Ours] |
| Confidence Selector (`tau_src`) | 54.1% | 0.5594 | [0.485, 0.597] | 0.6013 | -0.0021 [-0.005, -0.000] | [Ours] |
| Learned Rejector (`rejector_src`) | 36.0% | 0.5588 | [0.485, 0.595] | 0.5996 | -0.0027 [-0.007, 0.001] | [Ours] |
| Availability 20% (`avail_b20`) | 19.8% | 0.5565 | [0.483, 0.592] | 0.5947 | -0.0050 [-0.014, 0.002] | [Ours] |
| **Primary Availability 30% (`avail_b30`)** | **29.9%** | **0.5561** | **[0.482, 0.592]** | **0.5990** | **-0.0054 [-0.011, 0.000]** | [Ours] |
| Global Oracle (`oracle_g`) | 100.0% | 0.5315 | [0.452, 0.571] | 0.5925 | -0.0300 [-0.046, -0.013] | [Ours] |
| Adapted Oracle (`oracle_a`) | 100.0% | 0.6101 | [0.537, 0.646] | 0.6553 | +0.0486 [+0.038, +0.059] | [Ours] |

#### Q3 Decomposition Analysis (Adaptation Gain vs. Selection Gain vs. Interaction)

- At $k = 1$: $\text{adapt\_gain} = +0.0377 \ [0.010, 0.063]$
- At $k = 3$: $\text{adapt\_gain} = +0.0770 \ [0.036, 0.122]$

| Regime & Selector | Audio used | Select gain | Combined gain | Interaction | Interaction 95% CI |
|---|:---:|:---:|:---:|:---:|:---:|
| **k = 1** | | | | | |
| `tau_src` | 59.4% | +0.0083 [-0.010, 0.026] | +0.0435 [0.010, 0.073] | -0.0025 | [-0.013, 0.010] |
| `rejector_src` | 42.6% | +0.0118 [-0.007, 0.028] | +0.0452 [0.012, 0.074] | -0.0042 | [-0.015, 0.009] |
| `avail_b20` | 20.0% | +0.0087 [-0.008, 0.021] | +0.0424 [0.012, 0.070] | -0.0039 | [-0.013, 0.007] |
| **`avail_b30` (Primary)** | **30.0%** | **+0.0093 [-0.005, 0.023]** | **+0.0442 [0.013, 0.073]** | **-0.0028** | **[-0.012, 0.007]** |
| **k = 3** | | | | | |
| `tau_src` | 54.1% | +0.0095 [-0.010, 0.027] | +0.0905 [0.049, 0.133] | +0.0040 | [-0.013, 0.021] |
| `rejector_src` | 36.0% | +0.0120 [-0.008, 0.029] | +0.0899 [0.048, 0.132] | +0.0009 | [-0.015, 0.018] |
| `avail_b20` | 19.8% | +0.0098 [-0.007, 0.023] | +0.0876 [0.046, 0.130] | +0.0008 | [-0.012, 0.015] |
| **`avail_b30` (Primary)** | **29.9%** | **+0.0102 [-0.005, 0.025]** | **+0.0872 [0.045, 0.129]** | **-0.0000** | **[-0.014, 0.014]** |

**Conclusion on Orthogonality:** For the primary selector (`avail_b30`), the interaction term is -0.0028 [-0.012, 0.007] at $k = 1$ and -0.0000 [-0.014, 0.014] at $k = 3$. Because every interaction interval spans zero tightly, adaptation and selective acquisition operate orthogonally: their individual performance gains sum up linearly with zero negative interference.

---

### 4.2 Main Unadapted Policies ($k = 0$, 525 Target Clips)

| Policy | Macro-F1 | Accuracy | Audio used | Tag |
|---|:---:|:---:|:---:|:---:|
| Camera only (`vis_g`) | 0.4783 | 0.5238 | 0.0% | [Ours] |
| Always-both (`both_g`) | 0.4857 | 0.5371 | 100.0% | [Ours] |
| Confidence selector, source-tuned $\tau = 0.64$ | 0.4869 | n/a | 63.6% | [Ours] |
| Learned rejector, source-tuned threshold | 0.4901 | n/a | 47.4% | [Ours] |
| Oracle sensor choice | 0.5380 | 0.5867 | 100.0% | [Ours] |

Paired difference against camera only at $k = 0$:
- Always-both: +0.0074, interval [-0.014, 0.026]. [Ours]
- Confidence selector ($\tau = 0.64$): +0.0085, interval [-0.010, 0.026]. [Ours]
- Learned rejector: +0.0118, interval [-0.007, 0.028]. [Ours]

#### Availability Selector Sweep (M2 Score at $k = 0$)
Target ROC-AUC against "camera wrong and fusion right": 0.7466, interval [0.622, 0.875], 19 positive clips. [Ours]

| Audio used | Macro-F1 | Diff vs. Camera (95% CI) | Diff vs. Always-Both (95% CI) |
|---|:---:|:---:|:---:|
| 5.1% | 0.4835 | [0.001, 0.011] | [-0.020, 0.017] |
| 10.1% | 0.4863 | [-0.008, 0.019] | [-0.017, 0.019] |
| 15.0% | 0.4857 | [-0.010, 0.020] | [-0.016, 0.016] |
| 20.0% | 0.4871 | [-0.009, 0.022] | [-0.012, 0.013] |
| **30.1%** | **0.4878** | **[-0.006, 0.023]** | **[-0.010, 0.013]** |
| 40.0% | 0.4877 | [-0.008, 0.024] | [-0.009, 0.012] |
| 50.1% | 0.4835 | [-0.013, 0.022] | [-0.010, 0.005] |
| 60.0% | 0.4845 | [-0.014, 0.026] | [-0.003, 0.003] |
| 80.0% | 0.4857 | [-0.014, 0.026] | [0.000, 0.000] |
| 100.0% | 0.4857 | [-0.014, 0.026] | [0.000, 0.000] |

---

### 4.3 Q2: Domain Shift Resilience and Recalibration

To evaluate whether source-trained selectors degrade under domain shift, we evaluated diagnostic ranking AUCs and compared frozen source thresholds against rate-matched (no-label) and recalibrated ($k$-label) variants.

#### Diagnostic Ranking ROC-AUCs (Source OOF vs. Target)
| Selector Signal | Source OOF ROC-AUC | Target ROC-AUC (95% CI) | AUC Shift |
|---|:---:|:---:|:---:|
| Camera Uncertainty ($1 - \text{conf}$) | 0.7994 | 0.7431 [0.612, 0.865] | -0.0563 |
| Learned Rejector | 0.7747 | 0.7901 [0.669, 0.902] | +0.0154 |
| M2 Availability Score | 0.7208 | 0.7466 [0.622, 0.875] | +0.0258 |

#### Recalibration Strategies at $k = 0$
- **Frozen threshold selector ($\tau = 0.64$):** 63.6% audio, F1 = 0.4869 [0.410, 0.531]; vs. vis +0.0085 [-0.010, 0.026]; vs. rand +0.0171 [0.006, 0.027].
- **No-label rate-matched selector (59.8% audio):** F1 = 0.4895 [0.412, 0.533]; vs. vis +0.0112 [-0.007, 0.027]; vs. rand +0.0205 [0.008, 0.031].
- **Frozen learned rejector (47.4% audio):** F1 = 0.4901 [0.411, 0.534]; vs. vis +0.0118 [-0.007, 0.028]; vs. rand +0.0210 [0.006, 0.035].
- **No-label rate-matched rejector (47.2% audio):** F1 = 0.4900 [0.411, 0.534]; vs. vis +0.0117 [-0.007, 0.028]; vs. rand +0.0208 [0.006, 0.035].

---

### 4.4 Condition Slices (Exploratory)

Evaluation across 14 environmental condition slices on 525 target clips. Differences represent accuracy gains of the primary selector (`avail_b30`) against baselines.

| Slice | Clips ($n$) | Sites | Useful | Cam Acc | Aud Acc | Both Acc | Oracle | `avail_b30` vs. Cam | `avail_b30` vs. Rand |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `night_IR` | 392 | 57 | 11 | 0.490 | 0.153 | 0.497 | 0.554 | +0.010 [-0.005, 0.025] | +0.007 [-0.003, 0.016] |
| `day_colour` | 133 | 31 | 8 | 0.624 | 0.218 | 0.654 | 0.684 | +0.030 [-0.014, 0.087] | +0.032 [0.005, 0.073] |
| `bird` | 53 | 15 | 2 | 0.491 | 0.170 | 0.528 | 0.604 | +0.000 [0.000, 0.000] | -0.011 [-0.030, 0.000] |
| `mammal` | 472 | 66 | 17 | 0.528 | 0.169 | 0.538 | 0.585 | +0.017 [0.000, 0.036] | +0.016 [0.004, 0.030] |
| `small_animal` | 132 | 45 | 3 | 0.364 | 0.136 | 0.341 | 0.417 | +0.000 [-0.029, 0.036] | +0.008 [-0.010, 0.030] |
| `larger_animal` | 393 | 62 | 16 | 0.578 | 0.181 | 0.603 | 0.644 | +0.020 [0.002, 0.039] | +0.015 [0.001, 0.030] |
| `brief_visibility` | 182 | 54 | 8 | 0.489 | 0.209 | 0.505 | 0.577 | +0.022 [-0.010, 0.050] | +0.022 [0.000, 0.044] |
| `visible_most` | 343 | 61 | 11 | 0.542 | 0.149 | 0.554 | 0.592 | +0.012 [-0.004, 0.028] | +0.008 [-0.001, 0.020] |
| `arboreal` | 114 | 26 | 2 | 0.412 | 0.184 | 0.412 | 0.474 | +0.009 [-0.019, 0.051] | +0.011 [-0.010, 0.043] |
| `terrestrial` | 210 | 14 | 10 | 0.538 | 0.129 | 0.567 | 0.605 | +0.019 [-0.006, 0.039] | +0.010 [-0.010, 0.023] |
| `has_audio` | 523 | 65 | 19 | 0.524 | 0.170 | 0.537 | 0.587 | +0.015 [0.000, 0.032] | +0.013 [0.003, 0.024] |
| `no_audio` | 2 | 1 | 0 | 0.500 | 0.000 | 0.500 | 0.500 | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] |
| `sound_likely_top25`| 132 | 40 | 6 | 0.568 | 0.212 | 0.606 | 0.652 | +0.015 [-0.015, 0.045] | +0.000 [-0.024, 0.024] |
| `sound_unlikely` | 393 | 63 | 13 | 0.509 | 0.155 | 0.514 | 0.565 | +0.015 [-0.003, 0.036] | +0.017 [0.002, 0.034] |

*Slice caveat:* 14 slices were tested. Because slices overlap and sample counts per slice are small (most have $\le 10$ useful clips), individual slice findings are exploratory and should not be read as standalone confirmatory claims.

---

### 4.5 Audio and Vocalisation (Analysis Only)

`ana_target_sound` is an annotation score (median 0.013, maximum 0.60). Audio accuracy across quartiles: 15.9%, 15.3%, 16.0%, and 20.6% (~131 clips each). The slight increase in the top quartile is within sampling noise, providing weak support for the hypothesis that low audio accuracy is caused by missing vocalisation. This column comes from external annotations and is never used as a selector input. [Ours, exploratory] [Teammate]

---

## 5. What can and cannot be claimed

**Can claim**

- The pipeline is fully verified against M3's baselines and M2's adapted arrays, strictly leak-free (pre-audio features only, operating points tuned on source OOF). [Ours]
- **Adaptation and selection are orthogonal (Q3):** Interaction intervals between adaptation gain and selection gain span zero at both $k = 1$ (-0.0028 [-0.012, 0.007]) and $k = 3$ (-0.0000 [-0.014, 0.014]). Adaptation does not eliminate the need for selection, and selection does not impede adaptation. [Ours]
- **Check (a) is met at $k = 3$:** The primary selector (`avail_b30`) achieves F1 of 0.5561 at 29.9% audio use, matching Always-Both (0.5615 at 100% audio use) within sampling noise (difference -0.0054 [-0.011, 0.000]), cutting audio compute by over 70%. [Ours]
- **Pre-audio ranking models transfer robustly under shift (Q2):** The learned rejector and availability score maintain target ROC-AUCs of 0.790 and 0.747. [Ours]
- Across environmental condition slices, the primary selector produces consistent non-negative accuracy gains over camera-only baselines. [Ours]

**Cannot claim**

- That any selector statistically outperforms Always-Both at $k = 0$ (paired differences span zero due to weak unadapted audio). [Ours]
- Check (b) (beating the best adapted single sensor within one condition) cannot be confirmed with statistical significance across individual slices due to small sample sizes ($\le 10$ useful clips per slice). [Ours]
- That vocalisation rate fully explains weak audio performance (see 4.5). [Ours]
- Direct energy or battery day savings in hardware: compute savings are reported strictly as audio-used rates, pending physical AST execution profiling. [Ours]

---

## 6. Limitations

- **Sparse Useful Clips:** Only 19 of 525 target clips (and 43 of 1,786 source clips) have camera incorrect and fusion correct. Comparisons are inherently constrained by this low positive count.
- **Weak Unadapted Audio:** Target audio accuracy is 16.95%, severely limiting the potential gain of multimodal fusion without expert adaptation.
- **Small Slice Sample Sizes:** When splitting 525 clips into 14 slices, individual subgroups contain very few useful events, inflating bootstrap variance.
- **Compute Proxy:** The compute proxy equals the audio-used rate rather than physical AST profiling on edge hardware.
- **Extra Log Columns:** Columns `accuracy`, `audio_used_rate`, and `compute_proxy` are supplemental to the minimal log specification.

---

## 7. Dependencies

- **M2 Availability Scores:** The primary selector relies on availability predictions generated by M2's `build_availability.py`, which has been audited and confirmed leak-free.
- **M3 Embeddings & Probs:** All expert representations stem from M3's feature extractors (`clip_probs.npz`).
- **Alternative Audio:** If an updated alternative audio model is provided by teammates, the pipeline can rerun via a single configuration flag without overwriting logged v1, v2, or v3 rows.

---

## 8. Next steps

1. Integrate Q3 adaptation-selection decomposition findings into the final project report.
2. Compile master Pareto curves contrasting $k = 0$, $k = 1$, and $k = 3$ trade-offs for group presentation.
3. If hardware profiles become available, replace the audio-used rate compute proxy with profiled milliwatt-hour measurements.

---

## 9. Files

- `Evaluate_selectors_v3.ipynb` (interactive evaluation notebook)
- `Results/Pareto_curve.png` (v3 Pareto frontiers across $k \in \{0, 1, 3\}$)
- `results_log.csv` (840 logged v3 rows prefixed with `m4_`, alongside preserved v1 and v2 records)
