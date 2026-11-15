# Ranking evaluation: Member 2 audio selector

## Protocol

This retrospective evaluation uses **only source clips** and the pre-existing `availability.csv` source probabilities. Those probabilities are site-grouped OOF predictions: a clip's score came from a model trained without its `location_id` fold. No model was refit, tuned, thresholded, or calibrated for this analysis. Adapt and test clips, their labels, and all target outcomes are excluded.

The common candidate pool is 1785 source clips with an available audio stream (1786 source clips total; one source clip has no audio). A clip is useful exactly when the existing audio expert is correct and the existing camera expert is incorrect. At each nominal budget, every policy selects `ceil(budget x 1785)` candidates, so every comparison has exactly the same audio cost.

Policies: (1) **top-ranked**, descending usefulness OOF probability; (2) **camera uncertainty**, descending `1 - conf_camera`; (3) **random**, an independently shuffled fixed-size selection for seeds 0-99. `video_name` is merely a deterministic tie-breaker for the ranked policies, not a feature.

Precision is useful clips / processed clips. Recall is useful clips captured / all useful candidates. Audio usage is processed clips / eligible candidates. Audio-correction rate is useful clips captured / eligible candidates (the deployment-wide yield); it equals precision x audio usage. Random variability is population SD plus minimum and maximum across the 100 seed runs.

## Results

| Budget | Policy | Precision | Recall | Useful captured | Audio usage | Correction rate |
|---:|---|---:|---:|---:|---:|---:|
| 10% | top_ranked | 9.50% | 11.56% | 17.00 | 10.03% | 0.95% |
| 10% | camera_uncertainty | 14.53% | 17.69% | 26.00 | 10.03% | 1.46% |
| 10% | random_mean (mean +/- SD) | 8.12% +/- 1.81% | 9.89% +/- 2.20% | 14.54 +/- 3.24 | 10.03% +/- 0.00% | 0.81% +/- 0.18% |
| 20% | top_ranked | 10.64% | 25.85% | 38.00 | 20.00% | 2.13% |
| 20% | camera_uncertainty | 12.61% | 30.61% | 45.00 | 20.00% | 2.52% |
| 20% | random_mean (mean +/- SD) | 8.13% +/- 1.19% | 19.74% +/- 2.90% | 29.02 +/- 4.26 | 20.00% +/- 0.00% | 1.63% +/- 0.24% |
| 30% | top_ranked | 11.01% | 40.14% | 59.00 | 30.03% | 3.31% |
| 30% | camera_uncertainty | 13.06% | 47.62% | 70.00 | 30.03% | 3.92% |
| 30% | random_mean (mean +/- SD) | 8.10% +/- 0.97% | 29.54% +/- 3.55% | 43.42 +/- 5.23 | 30.03% +/- 0.00% | 2.43% +/- 0.29% |
| 50% | top_ranked | 10.86% | 65.99% | 97.00 | 50.03% | 5.43% |
| 50% | camera_uncertainty | 12.43% | 75.51% | 111.00 | 50.03% | 6.22% |
| 50% | random_mean (mean +/- SD) | 8.20% +/- 0.72% | 49.84% +/- 4.36% | 73.26 +/- 6.41 | 50.03% +/- 0.00% | 4.10% +/- 0.36% |

The per-seed random results are retained in `audio_selector_random_seeds.csv`; aggregate values and random min/max are in `audio_selector_budget_results.csv`.
