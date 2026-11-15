# Shim-inspired cost-sensitive learned audio rejector

## Data flow and scope

The experiment joins source clip metadata, camera posteriors, and visual embeddings by `video_name`. It uses 1785 audio-eligible source clips, 147 positive usefulness outcomes, and five disjoint site folds. The referenced `selectors/l2d.py` and `selectors/blanking.py` are absent from this checkout; existing Member 2 baselines were inspected but not changed. No Member 3 file is read or modified.

The selector receives only camera posterior/derived uncertainty, visual embedding, IR, and placement. Audio posterior/embedding, audio correctness, label/species, site/project, and annotations are excluded from selector features. Audio posteriors are used only after a selection is frozen to evaluate the resulting camera-or-audio classifier.

## Formulation

Shim et al. motivates deciding whether an additional feature/modality is worth acquisition cost. This implementation is a supervised approximation, not their RL, set encoder, or joint classifier-policy algorithm: it estimates `P(audio correct AND camera wrong | pre-audio camera information)` and requests audio when estimated benefit exceeds normalized cost. The initial target is retained exactly as specified. Its limitations are important: it ignores partial probability improvements, inherits both expert models’ errors, and treats a correct audio top-1 outcome as a unit benefit.

## Source grouped validation

The selected learned candidate is **random_forest / A_posterior** by source OOF AP among learned models: AP 0.1478 (site-bootstrap 95% 0.1146–0.1965). Camera uncertainty is the stronger reference: AP 0.1557 (0.1122–0.2149). Therefore the experiment does not claim a learned improvement over uncertainty. `feature_ablation_metrics.csv` and `fold_metrics.csv` contain all ablations/folds.

## Policy and cost outputs

`policy_comparison.csv` reports camera-only, audio-always, common-seed random, uncertainty, and learned policies at identical 10/20/30/50% budgets, including macro-F1, accuracy, log loss, calls, useful/unnecessary acquisitions, and normalized compute proxy (audio inference time is unavailable, so proxy equals acquisition rate). `cost_sensitivity.csv` uses low/medium/high coefficients 0.04/0.08/0.12 as sensitivity settings, not optimized values. `target_transfer.csv` applies the source-selected learned model separately to adaptation and test; target labels were not used in fitting, selection, thresholds, or calibration.

## Blanking

Blanking is kept distinct from acquisition: it tests robustness when selected audio is unavailable, whereas the rejector decides before audio processing. `blanking_results.csv` records uniform and context-dependent blanking at 0%, 30%, and 60%; this retrospective ablation does not train the rejector.

## Limitations

Only 147 positives and a finite number of sites make model comparisons unstable. Site bootstrap conditions on OOF scores and does not repeat full fitting. Context/visual features can retain site-correlated shortcuts despite grouped folds. Target results are evaluation only and cannot establish threshold transport.
