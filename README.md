# Adaptive Sensor Selection for Wildlife Monitoring under Domain Shift

Course project (AFML), team Gradient Drift. This branch holds the clip-level part on SA-FARI: the camera and audio experts, how much they drop under geographic shift, the shortcut test on audio, the few-label adapted experts, and the compute cost. The selectors and the station-level analysis are in the other members' parts.

## Question
At new camera sites, does adapting the expert with a few labels and choosing which sensor to use (camera or audio) solve the same problem, or different ones?

## Data
- **SA-FARI** (CVPR 2026, CC BY-NC): camera-trap videos with audio, with an official location-disjoint train/test split. Annotations are on Hugging Face (`facebook/SA-FARI`), videos on `gs://cxl-public-camera-trap/sa_fari/`. Raw videos and annotations are not in this repo.
- Our subset: 23 species.
  - Source: 1,786 clips at 184 sites, from the official train split, at most 80 clips per species, spread over sites.
  - Target: 525 clips at 66 new sites, from the official test split.
  - No site appears in both.

## Setup
- Python packages: `numpy pandas scikit-learn torch open_clip_torch transformers opencv-python pillow imageio-ffmpeg ijson matplotlib`
- `ffprobe` (from ffmpeg) is needed for `metadata_building.py`.
- Paths are set in `config.py`. Set the `SAFARI_ROOT` environment variable to your `safari/` folder.

Expected layout:
```
safari/
  sa_fari_train_ext.json, sa_fari_test_ext.json   (annotations)
  train_files.txt                                 (listing of the train video bucket)
  train_videos/, test_videos/                     (videos as <video_name>.mp4)
  outputs/                                        (everything the scripts write)
  outputs/results/                                (final result tables and figures)
```

The train videos used are listed in `safari/outputs/train_list.txt`. To download them:
```bash
gcloud storage cp --read-paths-from-stdin --no-clobber safari/train_videos/ < safari/outputs/train_list.txt
```

## Pipeline (run in this order)
All scripts are in `scripts/`. Run them from the project folder, e.g. `python scripts/make_split.py`.

**Data and split**
1. `metadata_building.py`: per-video metadata for the test set → `test_meta.csv` (uses `night_check.py` for infrared detection)
2. `train_index.py`: one row per train video from the large annotation file → `train_index.csv`
3. `make_split.py`: species subset and source/target split → `split.csv`, `train_list.txt`

**Features**

4. `extract_features.py`: BioCLIP (4-frame mean) and AST audio embeddings, AudioSet probabilities → `visual_emb.npy`, `audio_emb.npy`, `audioset_probs.npy`, `feat_index.csv`
5. `extract_frames_1fps.py`: BioCLIP embedding of every frame at 1 fps → `visual_frames_1fps.npz`
6. `source_conditions.py`: light (IR/colour) and placement for source clips → `source_meta.csv`

**Experts and Q1**

7. `baselines.py`: global camera and audio experts on the target → `results_summary.csv`, `results_per_species.csv`, `results_slices.csv`, `predictions.*`
8. `sanity_checks.py`: alignment, reproduction of the baselines, chance levels, shuffled labels (prints only)
9. `shift_levels.py`: Q1, drop from seen site → new site → new project → `shift_results.csv`, `shift_oof.csv`
10. `shift_drop_diff.py`: does audio drop more than the camera? → `shift_drop_diff.csv`

**Shortcut test**

11. `shortcut_test.py`: audio vs project-based guesses → `shortcut_results.csv`
12. `shortcut_night.py`: within project × light → `shortcut_night_results.csv`
13. `shortcut_placement.py`: within project × light × placement → `shortcut_placement_results.csv`
14. `audio_call_check.py`: is audio better on clips with a clear animal sound? → `audio_call_check*.csv`

**Expert probabilities, adaptation, cost**

15. `global_expert_probs.py`: expert probabilities for every clip (source out-of-fold by site) → `clip_events.csv`, `clip_probs.npz`
16. `adapted_experts.py --phase tune`, then `--phase target`: few-label adaptation (k = 1, 3 labelled target clips per species, seeds 0–4) → `adapt_weight_tuning.csv`, `adapted_results.csv`, `adapt_site_split.csv`, `adapt_draws.csv`, `adapted_probs.npz`
17. `compute_timing.py --device cuda` and `--device cpu`: time and FLOPs per clip → `compute_timing_*.csv`, `compute_summary_*.csv`

**Results**

18. `results_log.py`: all results in one table → `results_log_m3.csv` (+ `results_log.csv` with M1's rows when `m1_station_results.zip` is in the project folder, next to `scripts/`)

**Figures** are made by the script that computes the result and saved in `results/figures/` (PNG and PDF; shared style in `plot_utils.py`):
- `fig1_shift_levels`: `shift_levels.py`
- `fig2_adapt_decomp`, `fig3_headroom`: `adapted_experts.py --phase target`
- `fig4_cost`: `compute_timing.py` (shows every device timed so far)
- `fig5_shortcut`: `shortcut_placement.py` (needs the other two shortcut results)
- `fig6_m1_coverage`: `results_log.py` (only when M1's zip is present)
- `fig7_call_quintiles`: `audio_call_check.py`

Shared modules (also in `scripts/`): `plot_utils.py` (figure style), `config.py` (paths), `camera_expert.py` (camera model), `night_check.py` (IR detection).

Long runs (one camera fit takes about 3.5 min):
- `shift_levels.py` and `shortcut_test.py` take about 1 h each.
- `global_expert_probs.py` takes about 20–25 min.
- `adapted_experts.py` takes about 45 min per phase.

## Models
- **Camera:** frozen BioCLIP embeddings of frames at 1 fps, class-balanced logistic regression (C = 0.01), each clip given equal weight. The clip's prediction is its most confident frame.
- **Audio:** frozen AST embeddings (16 kHz mono, 10 s windows, averaged), class-balanced logistic regression. Clips without audio get uniform probabilities.
- All intervals are 95% bootstrap intervals that resample whole sites.

## Main results (target = new sites)
- The camera is far ahead: 52.4% accuracy vs 17.0% for audio (uniform guess 4.3%, most common species 12.8%).
- Both sensors lose accuracy as clips move further from the training data. The camera loses the most when moving to a new site. There is no evidence that audio drops more than the camera.
- Within the same project, light and placement, audio still carries some species information. On the target, though, it does not beat a guess based on project and time of day.
- Few-label adaptation (k = 3):
  - At cameras that got no labels, the camera improves by only +1.3 points.
  - For the same species at a labelled camera, it improves by +8.8 points.
  - So the labels mostly teach the model those cameras, not the region.
- The room for sensor selection (oracle minus camera) stays about the same after adapting: 6.3 vs 6.6 points. Adapting and selecting fix different errors, but both gains are small when the camera sees the animal.
- Cost: audio needs almost as many FLOPs as the camera, so always running both costs about 1.9× the camera alone.

Full numbers with intervals are in `safari/outputs/results/results_log.csv`.

## Files in `safari/outputs/`
- `split.csv`: video_name, species, location_id, split (source / adapt / test)
- `feat_index.csv`: row order of all feature and probability arrays
- `clip_events.csv`, `clip_probs.npz`: per-clip table and global expert probabilities, the input for the selectors. Columns starting with `ana_` are ground-truth analysis columns and must not be used as selector inputs.
- `adapted_probs.npz`, `adapt_site_split.csv`, `adapt_draws.csv`: adapted expert probabilities, the pool / held-out site split per seed, and the labelled clips per draw
- `results/`: final tables and `figures/`
- Not included because of size: `visual_frames_1fps.npz` (62 MB, made by `extract_frames_1fps.py`; needed to refit the camera expert).
