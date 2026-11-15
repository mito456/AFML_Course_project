# Shortcut control: audio vs the most common species within each project x light (IR/colour) group
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import camera_expert as cam
from config import OUT, RESULTS, SEED

N_BOOT = 1000
MIN_CLIPS, MIN_SPECIES, MIN_SITES = 10, 3, 4


def project(loc):
    l = str(loc)
    if l == "nan":
        return "unknown"
    if re.fullmatch(r"[0-9a-f]{8}", l):
        return "hash_site"
    m = re.match(r"[A-Za-z]+", l)
    p = m.group(0).upper() if m else "unknown"
    return "COLPA" if p == "COPLA" else p


def fit_predict(X_tr, y_tr, X_te):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def metrics(yt, p):
    return np.array([accuracy_score(yt, p),
                     f1_score(yt, p, labels=np.unique(yt), average="macro", zero_division=0)])


rows = []


def report(clip_set, yt, sites, preds, uniform, ref="audio"):
    uniq = np.unique(sites)
    members = {s: np.flatnonzero(sites == s) for s in uniq}
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in preds}
    for _ in range(N_BOOT):
        r = np.concatenate([members[s] for s in rng.choice(uniq, len(uniq))])
        for k, p in preds.items():
            draws[k].append(metrics(yt[r], p[r]))
    draws = {k: np.array(v) for k, v in draws.items()}
    print(f"\n--- {clip_set}: {len(yt)} clips, {len(uniq)} sites, {len(np.unique(yt))} species, "
          f"uniform chance {uniform:.3f} ---")
    out = {k: (metrics(yt, p), *np.percentile(draws[k], [2.5, 97.5], axis=0)) for k, p in preds.items()}
    for k in preds:
        if k != ref:
            d = draws[ref] - draws[k]
            out[f"{ref} - {k}"] = (metrics(yt, preds[ref]) - metrics(yt, preds[k]), *np.percentile(d, [2.5, 97.5], axis=0))
    for k, (pt, lo, hi) in out.items():
        print(f"{k:18s} acc {pt[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}]  macro-F1 {pt[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}]")
        for j, met in enumerate(["accuracy", "macro_f1"]):
            rows.append({"clip_set": clip_set, "predictor": k, "metric": met, "value": pt[j], "ci_lo": lo[j],
                         "ci_hi": hi[j], "n_clips": len(yt), "n_sites": len(uniq), "uniform": uniform})


idx = pd.read_csv(OUT / "feat_index.csv")
A = np.nan_to_num(np.load(OUT / "audio_emb.npy"))
sm = pd.read_csv(OUT / "source_meta.csv")[["video_name", "is_ir"]]
idx = idx.merge(sm, on="video_name", how="left")
idx["project"] = idx.location_id.map(project)
classes = sorted(idx.species.unique())
y, sites = idx.species.to_numpy(), idx.location_id.astype(str).to_numpy()
has_a = idx.has_audio.astype(bool).to_numpy()
frames = cam.load_frames(idx)

src = idx[idx.split == "source"].copy()
src["light"] = src.is_ir.astype(str).str.lower().map({"true": "IR", "false": "colour"})
print(f"source {len(src)} | IR {(src.light == 'IR').sum()} | colour {(src.light == 'colour').sum()} | "
      f"unknown {src.light.isna().sum()}")

# Site-grouped CV inside each group; a species needs >= 10 clips in the group
oof, n_uniform = [], []
for (p, light), g in src.dropna(subset=["light"]).groupby(["project", "light"]):
    vc = g.species.value_counts()
    keep = vc[vc >= MIN_CLIPS].index
    g = g[g.species.isin(keep)]
    if len(keep) < MIN_SPECIES or g.location_id.nunique() < MIN_SITES:
        continue
    ix = g.index.to_numpy()
    gs = sites[ix]
    out = {k: np.empty(len(ix), object) for k in ["audio", "camera", "majority"]}
    for a, b in GroupKFold(min(5, len(np.unique(gs)))).split(ix, groups=gs):
        tr_i, te_i = ix[a], ix[b]
        tr_a = tr_i[has_a[tr_i]]
        out["audio"][b] = fit_predict(A[tr_a], y[tr_a], A[te_i])
        out["audio"][b[~has_a[te_i]]] = classes[0]
        out["camera"][b] = cam.fit_predict(frames, y[tr_i], tr_i, te_i, classes)
        out["majority"][b] = pd.Series(y[tr_i]).value_counts().idxmax()
    name = f"{p} {light}"
    print(f"\n{name}: species kept: {', '.join(keep)}")
    report(name, y[ix], gs, out, 1 / len(keep))
    oof.append(pd.DataFrame({"i": ix, "uniform": 1 / len(keep), **out}))

if not oof:
    raise SystemExit("No (project, light) group meets the minimums.")
o = pd.concat(oof)
report("pooled groups", y[o.i.to_numpy()], sites[o.i.to_numpy()],
       {k: o[k].to_numpy() for k in ["audio", "camera", "majority"]}, o.uniform.mean())
# Pre-registered rule: audio carries species information if the pooled interval excludes 0
pooled = [r for r in rows if r["clip_set"] == "pooled groups" and r["predictor"] == "audio - majority"
          and r["metric"] == "accuracy"][0]
print(f"\nPre-registered rule (pooled accuracy, audio - majority CI excludes 0): "
      f"{'MET' if pooled['ci_lo'] > 0 else 'NOT MET'} ({pooled['value']:+.3f} [{pooled['ci_lo']:+.3f}, {pooled['ci_hi']:+.3f}])")

pd.DataFrame(rows).to_csv(RESULTS / "shortcut_night_results.csv", index=False)
print(f"Saved {RESULTS / 'shortcut_night_results.csv'}")
