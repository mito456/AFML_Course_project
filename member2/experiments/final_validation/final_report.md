# Final Member 2 validation

> **Superseded for numerical reporting (2026-11-15).** This report describes an earlier output state (including a 1,000-resample statement). Use `../../../../SAFARI_MEMBER2_FINAL_REPORT.md` and the current `results/` CSVs instead. Preserved for traceability.

## Audit and provenance

Validated 1785 source audio-eligible clips from 184 sites. Camera and audio posteriors were joined by `video_name`; every source row declares `prob_source=oof_site_cv`, and supplied outer folds have zero site overlap. Source audio posterior is used only to construct/evaluate the retrospective 95/5 fusion target. Selectors receive camera posterior/uncertainty, visual embedding, IR and placement only.

Earlier incremental and Shim experiments choose candidates using pooled source OOF results, then report that same pooled set. This is not label leakage into a held-out clip, but it can create model-selection optimism. This run corrects it visibly: each outer-fold candidate/target choice is made only from inner grouped OOF predictions on outer-training sites.

## Nested design

Five fixed site-disjoint outer folds are shared by every method. In each fold, feature-set/model/meaningful-benefit target choices are selected by 30% inner grouped-OOF yield. The frozen model is then fit on all outer-training sites and scored once on held-out sites. No outer label enters preprocessing, selection, calibration, thresholding, or cost choice.

## 30% source nested-OOF policy comparison

| Policy | Macro-F1 | Log loss | Calls | Total benefit | Benefit/call |
|---|---:|---:|---:|---:|---:|
| camera_only | 0.5065 | 1.5763 | 0 | 0.000 | 0.0000 |
| audio_always | 0.5066 | 1.5532 | 1785 | 41.144 | 0.0230 |
| random | 0.5061 | 1.5699 | 539 | 11.471 | 0.0213 |
| camera_uncertainty | 0.5071 | 1.5580 | 539 | 32.693 | 0.0607 |
| binary_usefulness | 0.5070 | 1.5619 | 539 | 25.692 | 0.0477 |
| shim_cost_sensitive | 0.5070 | 1.5619 | 539 | 25.692 | 0.0477 |
| continuous_incremental | 0.5075 | 1.5549 | 539 | 38.226 | 0.0709 |
| binary_incremental | 0.5070 | 1.5620 | 539 | 25.590 | 0.0475 |

Paired site-bootstrap intervals (1,000 resamples) are in `results/paired_site_bootstrap.csv`; they resample sites with all their clips and condition on the fixed nested OOF predictions. Thus they do not fully capture split/model-selection uncertainty and are not significance claims.

## Frozen target evaluation

`results/target_transfer.csv` applies the frozen source-selected continuous ridge separately to adaptation and test at 30%. Target labels are used only for these reported metrics, never fitting or selection. The adaptation cohort is especially small.

## Decision

Continuous incremental value has the strongest 30% point estimate, but the paired intervals must decide whether it reliably clears camera uncertainty. If its interval includes practically relevant harm or improvement is limited to log loss, retain camera uncertainty as the simpler reference; do not infer domain generalization from the small target sets.

## Limitations

Only a limited number of sites and positive/useful outcomes are available; audio experts are weak/noisy; fixed fusion is an evaluation assumption; visual/context inputs may retain site-correlated structure; and bootstrap conditional on OOF scores understates full training uncertainty.
