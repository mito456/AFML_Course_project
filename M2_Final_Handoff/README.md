# Member 2 final handoff

This self-contained handoff preserves original project files and final-validation outputs unchanged.

- `SAFARI/` contains Member 2's clip-level artifacts. `availability.csv` is the 2,311-clip handoff table for Member 4. `adapt_methods.py` provides k=0 ranking adjustment and guarded k=1/k=3 recalibration; run `python -m unittest discover -s SAFARI -p test_adapt_methods.py` from this folder. It needs only the Python standard library. For few-label use, pass `protocol_ids` the original repository's `safari/outputs/adapt_draws.csv`, then supply approved usefulness labels and a documented source-derived prior.
- `TAIWAN/` is a concise interpretation of authoritative Member 1 station-level evidence, not a copy of its implementation.
- `DOCUMENTATION/` contains the audit, Member 4 handoff notes, and two-dataset research summary.

Member 4 should consume the SA-FARI availability table and adaptation module only. Taiwan findings are complementary context and must not be combined with SA-FARI metrics or used by applying the SA-FARI functions directly.
