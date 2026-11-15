# Methodology

This experiment is inspired by the acquisition-cost objective in Shim et al.
(2018), but is not a reproduction. It has no reinforcement-learning agent,
variable-size set encoder, joint classifier-policy training, or target-label
selection. A supervised rejector estimates the existing Member 2 target:
`audio correct AND camera incorrect` from pre-audio features only.

The source evaluation uses supplied five site-disjoint folds. Scaling and all
model fitting happen within each training fold; reported source scores are OOF.
The candidate chosen among learned models is selected only by source OOF AP.
The camera-uncertainty reference is retained even if it wins.

At a fixed budget, eligible clips with the greatest score request audio. For a
cost setting, request when estimated usefulness probability exceeds the
normalized cost coefficient. Cost is a sensitivity parameter, not fitted on
target data. Classification metrics use camera posterior when audio is not
requested and audio posterior when it is requested; those posterior values are
strictly retrospective evaluation inputs, never rejector features.

Blanking is a separate missing-audio robustness intervention and must not be
interpreted as learned acquisition. The requested legacy selectors are absent
from this checkout, so no such implementation could be preserved or invoked.
