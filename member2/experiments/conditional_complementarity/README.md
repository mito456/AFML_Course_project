# Reproduction

From the repository root run:

```powershell
.venv-member2\Scripts\python.exe member2\experiments\conditional_complementarity\conditional_complementarity.py
```

Inputs are read-only: `safari/outputs/clip_probs.npz` and `clip_events.csv`. The script validates posterior/event keys and writes all deliverables in this directory. It requires NumPy and scikit-learn from `.venv-member2`.
