# Methodology

The experiment reads only existing source OOF camera/audio posteriors (`prob_source=oof_site_cv`) to make the retrospective incremental log-loss target. Rejector inputs are strictly pre-audio: camera posterior and deterministic uncertainty summaries, visual embedding, IR, and placement. Five supplied disjoint site folds provide source OOF scores; all preprocessing/model fits occur inside each outer training partition. Models are deliberately shallow/regularized. Audio posterior is used only in target construction and policy evaluation.

The primary decision uses 95/5 camera/audio fusion; 90/10 is sensitivity-only. Model/target selection is source-only by OOF total delta at a pre-specified 30% budget. Target adaptation/test are final source-fit evaluations only. Site-level paired bootstrap resamples locations, never individual clips.
