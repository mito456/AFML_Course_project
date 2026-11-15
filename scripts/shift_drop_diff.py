# Tests whether audio drops more than the camera between shift levels (audio drop - camera drop)
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import OUT, RESULTS, SEED

N_BOOT = 1000
SENSORS = ["camera", "audio"]
STEPS = [("L0_same_sites", "L1_new_sites"), ("L1_new_sites", "L2_new_project"),
         ("L0_same_sites", "L2_new_project")]


def metrics(yt, p):
    return np.array([accuracy_score(yt, p),
                     f1_score(yt, p, labels=np.unique(yt), average="macro", zero_division=0)])


oof = pd.read_csv(OUT / "shift_oof.csv")
d = oof[oof.covered_L2].reset_index(drop=True)
y = d.species.to_numpy()
levels = sorted({lvl for step in STEPS for lvl in step})
P = {(s, lvl): d[f"pred_{s}_{lvl}"].to_numpy() for s in SENSORS for lvl in levels}
print(f"clips: {len(d)} | sites: {d.location_id.nunique()} | projects: {d.project.nunique()}")


def drops(rows):
    m = {k: metrics(y[rows], p[rows]) for k, p in P.items()}
    out = {}
    for a, b in STEPS:
        cam, aud = m[("camera", a)] - m[("camera", b)], m[("audio", a)] - m[("audio", b)]
        out[(a, b)] = np.stack([cam, aud, aud - cam])
    return out


point = drops(np.arange(len(d)))
rows_out = []
# Resamples by site, then by project
for unit in ["location_id", "project"]:
    g = d[unit].astype(str).to_numpy()
    uniq = np.unique(g)
    members = {u: np.flatnonzero(g == u) for u in uniq}
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in point}
    for _ in range(N_BOOT):
        r = np.concatenate([members[u] for u in rng.choice(uniq, len(uniq))])
        for k, v in drops(r).items():
            draws[k].append(v)

    print(f"\n=== Resampling unit: {unit} ({len(uniq)} groups, {N_BOOT} draws) ===")
    for (a, b), v in point.items():
        arr = np.array(draws[(a, b)])
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        for j, metric in enumerate(["accuracy", "macro_f1"]):
            share = np.mean(arr[:, 2, j] > 0)
            print(f"{a[:2]}->{b[:2]} {metric:8s} | camera drop {v[0, j]:+.3f}  audio drop {v[1, j]:+.3f} | "
                  f"audio - camera {v[2, j]:+.3f} [{lo[2, j]:+.3f}, {hi[2, j]:+.3f}] | "
                  f"draws where audio drops more: {share:.2f}")
            rows_out.append({"unit": unit, "step": f"{a[:2]}->{b[:2]}", "metric": metric,
                             "camera_drop": v[0, j], "audio_drop": v[1, j], "diff": v[2, j],
                             "ci_lo": lo[2, j], "ci_hi": hi[2, j], "share_audio_more": share})

pd.DataFrame(rows_out).to_csv(RESULTS / "shift_drop_diff.csv", index=False)
print(f"\nSaved {RESULTS / 'shift_drop_diff.csv'}")
