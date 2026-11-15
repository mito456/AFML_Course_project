# Shortcut test: does audio recognise the species, or only where and when the clip was recorded?
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
# Minimums for a project to enter the within-project test
MIN_CLIPS = 10
MIN_SPECIES = 3
MIN_SITES = 4


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


# Scores each predictor and its difference from audio with site-bootstrap 95% intervals
def report(test, clip_set, yt, sites, preds, ref="audio"):
    uniq = np.unique(sites)
    members = {s: np.flatnonzero(sites == s) for s in uniq}
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in preds}
    for _ in range(N_BOOT):
        r = np.concatenate([members[s] for s in rng.choice(uniq, len(uniq))])
        for k, p in preds.items():
            draws[k].append(metrics(yt[r], p[r]))
    draws = {k: np.array(v) for k, v in draws.items()}
    print(f"\n--- {test} | {clip_set}: {len(yt)} clips, {len(uniq)} sites, {len(np.unique(yt))} species ---")
    for k, p in preds.items():
        pt = metrics(yt, p)
        lo, hi = np.percentile(draws[k], [2.5, 97.5], axis=0)
        print(f"{k:22s} acc {pt[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]  macro-F1 {pt[1]:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]")
        for j, met in enumerate(["accuracy", "macro_f1"]):
            rows.append({"test": test, "clip_set": clip_set, "predictor": k, "metric": met, "value": pt[j],
                         "ci_lo": lo[j], "ci_hi": hi[j], "n_clips": len(yt), "n_sites": len(uniq)})
    for k in preds:
        if k == ref:
            continue
        d = draws[ref] - draws[k]
        pt = metrics(yt, preds[ref]) - metrics(yt, preds[k])
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        print(f"  {ref} - {k:15s} acc {pt[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}]  "
              f"macro-F1 {pt[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}]")
        for j, met in enumerate(["accuracy", "macro_f1"]):
            rows.append({"test": test, "clip_set": clip_set, "predictor": f"{ref} - {k}", "metric": met,
                         "value": pt[j], "ci_lo": lo[j], "ci_hi": hi[j], "n_clips": len(yt), "n_sites": len(uniq)})


idx = pd.read_csv(OUT / "feat_index.csv")
V, A = np.load(OUT / "visual_emb.npy"), np.nan_to_num(np.load(OUT / "audio_emb.npy"))
idx["project"] = idx.location_id.map(project)
meta = pd.read_csv(OUT / "test_meta.csv")[["video_name", "is_ir"]]
idx = idx.merge(meta, on="video_name", how="left")
assert len(idx) == len(V) == len(A)
classes = sorted(idx.species.unique())
y, sites, proj = idx.species.to_numpy(), idx.location_id.astype(str).to_numpy(), idx.project.to_numpy()
has_a = idx.has_audio.astype(bool).to_numpy()
S = np.flatnonzero(idx.split == "source")
T = np.flatnonzero(idx.split.isin(["adapt", "test"]))
night = idx.is_ir.astype(str).str.lower().isin(["1", "1.0", "true"]).to_numpy()
print(f"source {len(S)} | target {len(T)} | target night (IR) {night[T].sum()} / day {(~night[T]).sum()}")

tr = S[has_a[S]]
p_audio = fit_predict(A[tr], y[tr], A[T])
p_audio[~has_a[T]] = classes[0]
frames = cam.load_frames(idx)
p_cam = cam.fit_predict(frames, y[S], S, T, classes)

# Project guess: most common source species in the clip's project
src = idx.iloc[S]
proj_top = src.groupby("project").species.agg(lambda x: x.value_counts().idxmax())
p_proj = idx.project.iloc[T].map(proj_top).fillna(src.species.value_counts().idxmax()).to_numpy()

# Project + night guess: most common species at other target sites of the same project and light
tgt = idx.iloc[T].reset_index(drop=True)
tgt["night"] = night[T]
p_pn = np.empty(len(tgt), object)
n_fallback = 0
for i, r in tgt.iterrows():
    other = tgt[(tgt.location_id != r.location_id) & (tgt.project == r.project)]
    same = other[other.night == r.night]
    pool = same if len(same) else other
    n_fallback += len(same) == 0
    p_pn[i] = pool.species.value_counts().idxmax() if len(pool) else p_proj[i]
print(f"project+night prior: {n_fallback} clips had no same-project, same-night clip at other sites (fallback)")

# (b) all target clips, (c2) night and day clips separately
yt, st = y[T], sites[T]
preds = {"audio": p_audio, "camera": p_cam, "project prior": p_proj, "project+night prior": p_pn}
report("b", "target all", yt, st, preds)
for name, m in [("target night", tgt.night.to_numpy()), ("target day", ~tgt.night.to_numpy())]:
    report("c2", name, yt[m], st[m], {k: v[m] for k, v in preds.items()})

print("\n=== (c1) within-project, site-grouped CV (source) ===")
# (c1) within each project: site-grouped CV, audio vs the fold's most common species
oof = []
for p in sorted(src.project.unique()):
    sp = src[src.project == p]
    keep = sp.species.value_counts()
    keep = keep[keep >= MIN_CLIPS].index
    sp = sp[sp.species.isin(keep)]
    if len(keep) < MIN_SPECIES or sp.location_id.nunique() < MIN_SITES:
        continue
    ix = S[np.flatnonzero((src.project == p).to_numpy() & src.species.isin(keep).to_numpy())]
    g = sites[ix]
    out = {"audio": np.empty(len(ix), object), "camera": np.empty(len(ix), object),
           "majority": np.empty(len(ix), object)}
    for a, b in GroupKFold(min(5, len(np.unique(g)))).split(ix, groups=g):
        tr_i, te_i = ix[a], ix[b]
        tr_a = tr_i[has_a[tr_i]]
        out["audio"][b] = fit_predict(A[tr_a], y[tr_a], A[te_i])
        out["audio"][b[~has_a[te_i]]] = classes[0]
        out["camera"][b] = cam.fit_predict(frames, y[tr_i], tr_i, te_i, classes)
        out["majority"][b] = pd.Series(y[tr_i]).value_counts().idxmax()
    print(f"{p}: {len(ix)} clips, {len(np.unique(g))} sites, species kept: {', '.join(keep)}")
    report("c1", f"project {p}", y[ix], g, out)
    oof.append(pd.DataFrame({"i": ix, **out}))

o = pd.concat(oof)
report("c1", "pooled projects", y[o.i.to_numpy()], sites[o.i.to_numpy()],
       {k: o[k].to_numpy() for k in ["audio", "camera", "majority"]})

pd.DataFrame(rows).to_csv(RESULTS / "shortcut_results.csv", index=False)
print(f"\nSaved {RESULTS / 'shortcut_results.csv'}")
