# Task: Final Member 2 Validation — Nested Site-Grouped Comparison of Audio Acquisition Methods

## 1. Objective

We have completed three major Member 2 approaches for deciding whether audio should be acquired in wildlife monitoring.

### Approach A: Binary usefulness

Original definition:

`useful_audio = audio_correct AND camera_incorrect`

### Approach B: Cost-sensitive learned rejector

A supervised rejector inspired by Shim et al. (2018), estimating whether acquiring audio is worth its computational cost.

### Approach C: Incremental predictive value

Instead of binary top-1 correctness, estimate the improvement in predictive quality obtained from acquiring audio.

The incremental target is:

\[
\Delta_i =
L(y_i,p_{\text{camera},i})
-
L(y_i,p_{\text{camera+audio},i})
\]

Positive delta means audio reduces log loss.

Our preliminary results suggest that incremental predictive value is the most promising formulation.

At a 30% acquisition budget:

| Method | Total incremental benefit | Mean benefit/call |
|---|---:|---:|
| Camera uncertainty | 29.77 | 0.05555 |
| Continuous incremental | 37.06 | 0.06914 |
| Binary incremental | 39.98 | 0.07458 |

However, paired site-bootstrap intervals cross zero.

We therefore cannot claim that the new approaches genuinely outperform camera uncertainty.

### Final research question

**Can we predict the expected improvement in classification quality from acquiring audio, using only information available before audio processing?**

Our objective is to conduct a rigorous final comparison and freeze the Member 2 methodology.

Do not develop additional acquisition heuristics.

---

# 2. Inspect existing implementations first

Inspect the following existing directories:

- `member2/experiments/`
- `member2/experiments/conditional_complementarity/`
- `member2/experiments/shim_inspired_rejector/`
- `member2/experiments/incremental_audio_value/`
- Existing Member 2 selector implementations.
- M3 prediction and embedding files.

Read the existing reports, methodology documents, scripts, and result CSVs.

Understand the exact data flow and experimental assumptions.

Do not blindly trust previous reported metrics. Reproduce the important results where necessary.

Do not overwrite or modify existing experiments.

Create a new directory:

`member2/experiments/final_validation/`

All new outputs must be contained there.

---

# 3. Critical audit: Prediction provenance and leakage

Before conducting model selection, audit the provenance of all predictions and targets.

Verify:

1. Source camera posteriors are genuinely site-OOF.
2. Source audio posteriors are genuinely site-OOF.
3. Camera and audio predictions correspond to the same clips.
4. All joins use `video_name`.
5. No source site occurs in both training and validation folds.
6. No target labels are used for fitting, calibration, model selection, or threshold selection.
7. The incremental-value target is constructed using valid OOF expert predictions.
8. No audio-derived information enters the pre-audio rejector.

Pay special attention to the relationship between:

- Outer validation folds.
- Inner training folds.
- Construction of incremental-value targets.
- Model selection.
- Calibration.
- Acquisition threshold selection.

If any previous result contains leakage, identify it and explain its effect.

Do not silently correct methodological problems without documenting them.

---

# 4. Implement nested site-grouped validation

This is the most important part of the task.

Our earlier experiments compared multiple candidate models using source OOF results.

Selecting the best candidate and reporting performance on the same OOF results can introduce model-selection optimism.

Implement nested site-grouped validation.

### Outer loop

Use five site-grouped outer folds.

Each outer fold must represent unseen sites.

The outer validation fold must not influence:

- Model selection.
- Hyperparameter tuning.
- Feature selection.
- Calibration.
- Threshold selection.
- Cost selection.
- Acquisition-policy selection.

### Inner loop

Within each outer training partition:

- Perform grouped cross-validation.
- Select the model and hyperparameters.
- Fit preprocessing.
- Calibrate if needed.
- Select the acquisition threshold or policy parameters.
- Use only the outer training sites.

Then evaluate the frozen selected policy on the untouched outer validation sites.

Aggregate the outer-fold predictions to obtain final nested OOF results.

All candidate methods must use identical outer folds.

Document exactly where every fitting and selection operation occurs.

---

# 5. Compare the three usefulness formulations

We need a fair comparison.

## A. Binary top-1 usefulness

\[
z_i =
1 \quad \text{if audio is correct and camera is incorrect}
\]

Estimate:

\[
P(z_i=1\mid x_i)
\]

## B. Cost-sensitive learned rejector

Use the existing Shim-inspired supervised rejector.

Preserve its distinction from the original reinforcement-learning paper.

## C. Incremental predictive value

Primary target:

\[
\Delta_i =
\log p_{\text{fused},i}(y_i)
-
\log p_{\text{camera},i}(y_i)
\]

Positive delta means acquiring audio improves log loss.

Compare:

1. Continuous regression of delta.
2. Binary classification of delta > 0.
3. Meaningful-benefit classification using the previously specified thresholds.

Use the fixed 95/5 camera/audio fusion as the primary target construction.

Keep 90/10 as a sensitivity analysis only.

Do not tune fusion weights using outer validation or target labels.

---

# 6. Pre-audio features

All rejectors must use only information available before audio processing.

Permitted inputs:

- Camera posterior probabilities.
- Camera maximum confidence.
- Camera entropy.
- Camera top-two margin.
- Visual embedding.
- IR indicator.
- Permitted placement metadata.

Run a small, consistent feature ablation:

1. Camera posterior only.
2. Camera posterior + uncertainty.
3. Camera information + visual embedding.
4. Camera information + permitted environmental context.

Never include:

- Audio posterior.
- Audio embedding.
- Audio confidence.
- Audio correctness.
- Incremental-value target.
- True species.
- Site ID.
- Project ID.
- Oracle annotations.
- Audio-derived information.

Use the same feature availability rules across all three approaches.

---

# 7. Fair acquisition-policy comparison

Compare these methods:

1. Camera-only.
2. Audio-always.
3. Random acquisition.
4. Camera uncertainty.
5. Binary usefulness rejector.
6. Shim-inspired cost-sensitive rejector.
7. Continuous incremental-value rejector.
8. Binary incremental-value rejector.

Use matched acquisition budgets:

- 10%.
- 20%.
- 30%.
- 50%.

For every method calculate:

- Macro-F1.
- Accuracy.
- Log loss.
- Multiclass Brier score.
- Audio acquisition rate.
- Number of audio calls.
- Total incremental log-loss benefit.
- Mean incremental benefit per audio call.
- Number of harmful acquisitions.
- Number of beneficial acquisitions.
- Normalized compute proxy.

Use common random seeds for random acquisition.

All methods must be evaluated on the same outer validation clips.

Do not select a different best model separately on the outer validation fold.

---

# 8. Cost sensitivity

Evaluate a small set of predefined normalized audio costs.

For incremental-value methods:

\[
\text{Acquire audio if }
\widehat{\Delta}_i > \lambda C_{\text{audio}}
\]

For binary approaches, use an explicitly documented expected-benefit threshold.

Do not optimize cost coefficients using outer validation or target labels.

Report the complete trade-off between:

- Audio acquisition rate.
- Macro-F1.
- Log loss.
- Total incremental benefit.
- Computational cost.

Do not assume the highest-performing policy is automatically the best deployment policy.

---

# 9. Statistical comparison

Use paired site-cluster bootstrap confidence intervals with at least 1,000 resamples.

Calculate intervals for:

- Macro-F1 differences.
- Log-loss differences.
- Total incremental benefit.
- Mean incremental benefit per call.
- Acquisition-rate differences.

Compare the incremental-value approaches directly against camera uncertainty.

Also compare the best incremental-value approach against the original binary usefulness rejector.

Important:

- Resample sites, not individual clips.
- Preserve all clips belonging to a selected site.
- Use paired resampling for competing policies.
- Report the number of sites and clips.
- Explain that bootstrapping fixed OOF predictions does not capture all training and model-selection uncertainty.

Avoid claiming significance from overlapping or non-overlapping intervals alone. Report the paired difference intervals directly.

---

# 10. Final target evaluation

After completing source-only nested validation:

1. Select the methodology using source data only.
2. Fit the selected approach using permitted source information.
3. Apply it to target adaptation and target test separately.
4. Freeze all source-selected parameters and thresholds.
5. Do not use target labels for any model-selection decision.

Report:

- Target adaptation acquisition rate and metrics.
- Target test acquisition rate and metrics.
- Camera-only comparison.
- Audio acquisition cost.
- Incremental predictive benefit.

Clearly state the small sample sizes and limitations.

Do not claim domain generalization from weak or uncertain results.

---

# 11. Final decision framework

The experiment must answer these questions:

### Question 1
Does incremental predictive value provide a better acquisition signal than binary top-1 usefulness?

### Question 2
Does continuous regression outperform binary incremental-value classification?

### Question 3
Does any learned approach outperform camera uncertainty under nested site-grouped evaluation?

### Question 4
Are improvements present in log loss only, or also in macro-F1?

### Question 5
Does the best approach transfer to target sites?

### Question 6
Is any improvement large enough to justify the additional complexity and computational cost?

If no learned method beats camera uncertainty reliably, recommend retaining camera uncertainty as the simpler reference policy.

Do not force the incremental-value approach to win simply because its preliminary results were better.

---

# 12. Freeze Member 2 methodology

Once the final validation is complete, create a clear handoff.

Create:

- `final_validation.py`
- `nested_validation.py`
- `policy_comparison.py`
- `results/`
- `final_report.md`
- `final_methodology.md`
- `member2_handoff.md`
- `README.md`

The handoff must include:

### A. Research question
What exactly Member 2 investigated.

### B. Three formulations
Explain binary usefulness, cost-sensitive rejection, and incremental predictive value.

### C. Final experimental methodology
Explain nested site-grouped validation and leakage prevention.

### D. Results
Include complete comparisons and uncertainty intervals.

### E. Final recommendation
State which approach should be retained and why.

### F. Limitations
Explain data limitations, weak audio predictions, small positive sample counts, site variation, and prediction uncertainty.

### G. Integration guidance
Explain what Member 4 needs from Member 2 to integrate the final selector into the broader project.

Clearly separate:
- Proven findings.
- Suggestive findings.
- Unresolved hypotheses.

---

# 13. Freeze rules

This is a final validation experiment, not another exploratory model-development cycle.

Do not:

- Invent new acquisition heuristics.
- Add complex architectures without justification.
- Implement the complete Shim et al. RL framework.
- Change the original dataset splits.
- Modify Member 3 files.
- Modify existing Member 2 experiments.
- Tune on target labels.
- Force a positive result.

If nested validation reveals that the earlier incremental-value results were optimistic, report that honestly.

If camera uncertainty remains the strongest method, freeze it as the baseline rather than continuing to develop increasingly complicated alternatives.

Before coding, provide a short audit of the existing experiments and explain the planned nested-validation structure.

Then implement, execute, validate all deliverables, and produce the final Member 2 handoff.