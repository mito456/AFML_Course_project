# SA-FARI Member 2 final report

**Documentation identifier:** `SAFARI_MEMBER2_FINAL_VALIDATION_RECONCILIATION_2026-11-15` (documentation reconciliation, not an experimental run).  
**Authoritative result set:** `member2/experiments/final_validation/results/`, generated 2026-11-15 10:46--11:02 UTC.  
**Revision inspected:** `acccce7c7b86bd90f45aac5196908e2135eb1e75`.

## Scope and protocol

This report concerns SA-FARI only.  The decision unit is a camera-triggered clip for which audio is available; camera is always available and the decision is whether to process audio.  Cost is the fraction of eligible clips processed, not recorder-on time.  The source evaluation has 1,785 clips from 184 sites, uses supplied site-disjoint outer folds and source-only grouped inner OOF selection.  The target policy is selected on source data, fitted once on all source clips, and then frozen; target labels are evaluation-only.

Incremental benefit is the change in true-class log probability from the fixed 95/5 camera/audio fusion versus camera-only.  `camera_uncertainty` is the simple reference.  Oracle ranks realized benefit and is retrospective only; it is not deployable. `shim_cost_sensitive` is operationally identical to `binary_usefulness` in this run.

## Authoritative equal-budget comparison

The following Markdown table was regenerated from `member2/experiments/final_validation/results/policy_comparison.csv`; values are not copied from earlier reports. `Benefit` is total incremental benefit.

| Budget | Policy | Audio calls | Rate | Macro-F1 | Log loss | Benefit |
|---:|---|---:|---:|---:|---:|---:|
| 10% | camera_only | 0 | 0.0% | 0.5065 | 1.5763 | 0.000 |
| 10% | audio_always | 1785 | 100.0% | 0.5066 | 1.5532 | 41.144 |
| 10% | random | 180 | 10.1% | 0.5065 | 1.5746 | 2.969 |
| 10% | camera_uncertainty | 180 | 10.1% | 0.5075 | 1.5691 | 12.833 |
| 10% | binary_usefulness | 180 | 10.1% | 0.5075 | 1.5693 | 12.498 |
| 10% | shim_cost_sensitive | 180 | 10.1% | 0.5075 | 1.5693 | 12.498 |
| 10% | continuous_incremental | 180 | 10.1% | 0.5065 | 1.5685 | 13.881 |
| 10% | binary_incremental | 180 | 10.1% | 0.5070 | 1.5674 | 15.850 |
| 10% | oracle | 180 | 10.1% | 0.5075 | 1.5247 | 92.002 |
| 20% | camera_only | 0 | 0.0% | 0.5065 | 1.5763 | 0.000 |
| 20% | audio_always | 1785 | 100.0% | 0.5066 | 1.5532 | 41.144 |
| 20% | random | 360 | 20.2% | 0.5061 | 1.5727 | 6.480 |
| 20% | camera_uncertainty | 360 | 20.2% | 0.5075 | 1.5651 | 19.899 |
| 20% | binary_usefulness | 360 | 20.2% | 0.5070 | 1.5649 | 20.401 |
| 20% | shim_cost_sensitive | 360 | 20.2% | 0.5070 | 1.5649 | 20.401 |
| 20% | continuous_incremental | 360 | 20.2% | 0.5070 | 1.5599 | 29.304 |
| 20% | binary_incremental | 360 | 20.2% | 0.5070 | 1.5630 | 23.777 |
| 20% | oracle | 360 | 20.2% | 0.5075 | 1.5201 | 100.331 |
| 30% | camera_only | 0 | 0.0% | 0.5065 | 1.5763 | 0.000 |
| 30% | audio_always | 1785 | 100.0% | 0.5066 | 1.5532 | 41.144 |
| 30% | random | 539 | 30.2% | 0.5061 | 1.5699 | 11.471 |
| 30% | camera_uncertainty | 539 | 30.2% | 0.5071 | 1.5580 | 32.693 |
| 30% | binary_usefulness | 539 | 30.2% | 0.5070 | 1.5619 | 25.692 |
| 30% | shim_cost_sensitive | 539 | 30.2% | 0.5070 | 1.5619 | 25.692 |
| 30% | continuous_incremental | 539 | 30.2% | 0.5075 | 1.5556 | 36.886 |
| 30% | binary_incremental | 539 | 30.2% | 0.5066 | 1.5592 | 30.483 |
| 30% | oracle | 539 | 30.2% | 0.5075 | 1.5206 | 99.450 |
| 50% | camera_only | 0 | 0.0% | 0.5065 | 1.5763 | 0.000 |
| 50% | audio_always | 1785 | 100.0% | 0.5066 | 1.5532 | 41.144 |
| 50% | random | 894 | 50.1% | 0.5056 | 1.5648 | 20.579 |
| 50% | camera_uncertainty | 894 | 50.1% | 0.5071 | 1.5483 | 49.874 |
| 50% | binary_usefulness | 894 | 50.1% | 0.5066 | 1.5516 | 44.161 |
| 50% | shim_cost_sensitive | 894 | 50.1% | 0.5066 | 1.5516 | 44.161 |
| 50% | continuous_incremental | 894 | 50.1% | 0.5071 | 1.5531 | 41.415 |
| 50% | binary_incremental | 894 | 50.1% | 0.5066 | 1.5526 | 42.308 |
| 50% | oracle | 894 | 50.1% | 0.5075 | 1.5280 | 86.136 |

At the primary 30% point, continuous incremental value has the larger total-benefit point estimate than camera uncertainty (36.886 vs 32.693).

## Uncertainty and target transfer

`paired_site_bootstrap.csv` uses paired whole-site percentile bootstrap resampling: 2,000 replicates, seed `20261115`, 184 sites, and 2,000 valid samples for each cited comparison. For **continuous incremental minus camera uncertainty** at 30%, the 95% interval is -5.166 to 18.146 for total incremental benefit. It crosses zero. The same comparison has macro-F1 interval 0.0000 to 0.0013 and log-loss difference interval -0.0100 to 0.0030 (difference direction is continuous minus uncertainty; lower log loss favors continuous). Thus the higher benefit point estimate does **not** establish superiority over camera uncertainty. These conditional intervals do not include repeated fitting, split selection, or model-selection uncertainty.

The selected transfer policy in `source_selected_transfer_policy.csv` is a ridge continuous-incremental selector using `posterior_uncertainty`, selected at budget 0.30 from `source_only_inner_grouped_oof`, then fit on all eligible source clips. `target_transfer.csv` reports frozen evaluation: adaptation (69 clips; 21 calls/30.4%; F1 0.4609; log loss 1.6690; benefit 1.677) and test (454 clips; 137 calls/30.2%; F1 0.4800; log loss 1.6299; benefit 5.159). Camera-only values are included in that CSV. These mixed, small-cohort results do not establish convincing generalization and involve no target feedback, refitting, thresholding, or selection.

## Limitations and conclusion

This supports a source-protocol finding that pre-audio ranking can prioritize retrospective fusion benefit, not a general claim of adaptive recording. Camera-triggered sampling cannot evaluate events that the camera never detected. Audio experts and fixed late fusion are imperfect assumptions. The embedding mapping is verified at manifest level but cannot be independently reconstructed because the checkout lacks upstream embedding-generation code and embedded NPY identifiers. Oracle values are retrospective upper bounds, not deployment performance.

The result files above were produced by the documented `final_validation.py` / `nested_validation.py` / `policy_comparison.py` workflow (README command: `.venv-member2\\Scripts\\python.exe final_validation.py` from the final-validation directory). No command was run during this reconciliation.
