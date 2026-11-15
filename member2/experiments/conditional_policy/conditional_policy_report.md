# Conditional uncertainty policy experiment

## Design

Source-only evaluation used 1,785 audio-eligible clips and 147 useful clips, in the supplied five disjoint site-grouped outer folds. E1 is global `1 - max(camera posterior)` uncertainty; E2 maps five outer-training quantile bins to smoothed usefulness rates; E3/E4 add bin-by-IR or bin-by-placement cells; E5 uses their three-way cells. No species, site, project, target/adaptation data, or oracle feature entered a selector.

Every policy's outer-fold score was Platt-calibrated from inner site-grouped OOF predictions. Thus pooled comparisons use symmetric calibrated scores. `E1_uncertainty/raw_ranking` is separately reported only to distinguish the original monotonic uncertainty ranking from its calibrated global ranking; it is not an additional tuned method. The fixed five-pseudo-observation shrinkage is a stability rule, not label-selected tuning.

## Pooled calibrated budget yield

| Policy | Budget | Calls | Useful | Precision | Recall | Correction rate |
|---|---:|---:|---:|---:|---:|---:|
| E1_uncertainty | 10% | 179 | 26 | 0.145 | 0.177 | 0.015 |
| E1_uncertainty | 20% | 357 | 45 | 0.126 | 0.306 | 0.025 |
| E1_uncertainty | 30% | 536 | 70 | 0.131 | 0.476 | 0.039 |
| E1_uncertainty | 50% | 893 | 112 | 0.125 | 0.762 | 0.063 |
| E2_bins | 10% | 179 | 19 | 0.106 | 0.129 | 0.011 |
| E2_bins | 20% | 357 | 35 | 0.098 | 0.238 | 0.020 |
| E2_bins | 30% | 536 | 63 | 0.118 | 0.429 | 0.035 |
| E2_bins | 50% | 893 | 111 | 0.124 | 0.755 | 0.062 |
| E3_bins_x_ir | 10% | 179 | 15 | 0.084 | 0.102 | 0.008 |
| E3_bins_x_ir | 20% | 357 | 38 | 0.106 | 0.259 | 0.021 |
| E3_bins_x_ir | 30% | 536 | 64 | 0.119 | 0.435 | 0.036 |
| E3_bins_x_ir | 50% | 893 | 110 | 0.123 | 0.748 | 0.062 |
| E4_bins_x_placement | 10% | 179 | 20 | 0.112 | 0.136 | 0.011 |
| E4_bins_x_placement | 20% | 357 | 45 | 0.126 | 0.306 | 0.025 |
| E4_bins_x_placement | 30% | 536 | 64 | 0.119 | 0.435 | 0.036 |
| E4_bins_x_placement | 50% | 893 | 114 | 0.128 | 0.776 | 0.064 |
| E5_bins_x_ir_x_placement | 10% | 179 | 17 | 0.095 | 0.116 | 0.010 |
| E5_bins_x_ir_x_placement | 20% | 357 | 34 | 0.095 | 0.231 | 0.019 |
| E5_bins_x_ir_x_placement | 30% | 536 | 58 | 0.108 | 0.395 | 0.032 |
| E5_bins_x_ir_x_placement | 50% | 893 | 113 | 0.127 | 0.769 | 0.063 |

Raw fold-level yields and paired differences are in `fold_results.csv`; 100 common random permutations are in `conditional_policy_results.csv`. `budget_yield_intervals.csv` gives site-resampled 95% intervals for useful clips captured, rather than treating dependent clips as independent.

## Interaction checks and conclusion

The saved tables expose the bin-by-IR, bin-by-placement, and three-way policies directly. With only 147 positives, five outer folds are too few for small pooled differences to be persuasive: interpret a conditional policy as convincing only when its paired fold differences are consistently positive and its site-aware interval is materially separated from E1. On this deliberately low-dimensional analysis, the report does not claim a conditional improvement merely from a small numerical advantage; absent that consistent evidence, the simpler global uncertainty policy remains preferable.
