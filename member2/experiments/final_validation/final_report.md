# Final Member 2 validation

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

The source-selected continuous ridge was fitted once on source and applied unchanged at the 30% budget. On adaptation (69 clips), log loss was 1.6690 versus 1.6933 camera-only, with 1.677 total benefit; on test (454 clips), it was 1.6299 versus 1.6412, with 5.159 total benefit. Macro-F1 was unchanged in both cohorts. See `results/target_transfer.csv`. Target labels were evaluation-only; these small, uncertain results do not establish domain generalization.

## Decision

**Final recommendation:** retain camera uncertainty as the deployment reference. Continuous incremental value exceeds binary usefulness in total benefit (paired 95% interval 2.12 to 23.83), but it does not reliably beat camera uncertainty: the total-benefit interval is -6.04 to 18.40 and the log-loss-difference interval is -0.0099 to 0.0036. Macro-F1 is essentially unchanged. Treat continuous incremental value as suggestive only, not an established improvement or domain-generalization result.

Use the tables and paired intervals—not point estimates alone—to decide whether a learned selector clears camera uncertainty. If its interval includes practically relevant harm or improvement is limited to log loss, retain camera uncertainty as the simpler reference. This report deliberately makes no domain-generalization claim: target labels were not used, but a fully frozen target evaluation requires compatible target posterior/provenance handling beyond this source validation.

## Limitations

Only a limited number of sites and positive/useful outcomes are available; audio experts are weak/noisy; fixed fusion is an evaluation assumption; visual/context inputs may retain site-correlated structure; and bootstrap conditional on OOF scores understates full training uncertainty.
