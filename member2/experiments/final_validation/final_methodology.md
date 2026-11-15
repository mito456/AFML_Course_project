# Frozen methodology

Primary target is `log p(0.95 camera + 0.05 audio at true class) - log p(camera at true class)`. 90/10 is sensitivity-only and is not selected. The policy uses only pre-audio information. Nested five-fold site-grouped validation selects candidates within each outer training split; outputs are outer OOF only. Costs `.001, .005, .01` are predefined.
