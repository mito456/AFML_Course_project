# Member 2 research summary: two complementary datasets

The shared question is when an additional acoustic acquisition is worthwhile after camera information is available. The datasets address that question at different decision units and their metrics must not be pooled into one score.

## SA-FARI: clip-level camera/audio acquisition

SA-FARI makes a decision for each camera-triggered clip: whether audio processing is useful for that clip. The frozen camera-based selector is evaluated for classification benefit. Member 2 supplies validated availability scores, a no-label rank adjustment, and protocol-guarded few-label recalibration. Species traits are biological context only; they are not selector inputs or model-derived priors. k=0 adjusts ranks only, while k=1/3 needs approved adaptation IDs, their externally acquired usefulness labels, and an explicit source-derived prior.

## Taiwan Grass-Owl: station-hour camera/recorder acquisition

Member 1's authoritative Taiwan analysis asks whether recording is worthwhile in a station-hour. It evaluates detection capture, recorder-on rate, fixed and adaptive schedules, source-to-target river transfer, and station-bootstrap uncertainty. The adaptive policy receives target feedback through nightly updating and camera tip-offs; it is not the SA-FARI selector and must not be replaced by `adapt_methods.py`.

## Combined reading and limits

Taiwan shows substantial recorder-only detections, while SA-FARI examines conditional clip-level classification value. Together they support studying adaptive acoustic acquisition across complementary sensor footprints, but do not establish a common effect size, calibrated transfer across datasets, or a universal policy. Taiwan has 13 stations in two river systems and footprint/geography limitations; SA-FARI's clip-level labels and species context have different sampling and endpoint assumptions.
