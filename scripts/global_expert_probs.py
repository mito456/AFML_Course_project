# Global expert probabilities for every clip -> clip_events.csv and clip_probs.npz (inputs to the selectors)
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import camera_expert as cam
from config import OUT

MIN_TRAIN_L2 = 5


def project(loc):
    l = str(loc)
    if l == "nan":
        return "unknown"
    if re.fullmatch(r"[0-9a-f]{8}", l):
        return "hash_site"
    m = re.match(r"[A-Za-z]+", l)
    p = m.group(0).upper() if m else "unknown"
    return "COLPA" if p == "COPLA" else p


# Audio expert; clips without audio get uniform probabilities
def audio_proba(tr, te):
    tr = tr[has_a[tr]]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    m.fit(A[tr], y[tr])
    p = np.zeros((len(te), len(classes)))
    p[:, [classes.index(c) for c in m.classes_]] = m.predict_proba(A[te])
    p[~has_a[te]] = 1.0 / len(classes)
    return p


def experts(tr, te):
    return cam.fit_proba(frames, y[tr], tr, te, classes), audio_proba(tr, te)


# --with-l2 also saves leave-one-project-out probabilities for the source clips
ap = argparse.ArgumentParser()
ap.add_argument("--with-l2", action="store_true")
args = ap.parse_args()

idx = pd.read_csv(OUT / "feat_index.csv")
A = np.nan_to_num(np.load(OUT / "audio_emb.npy"))
frames = cam.load_frames(idx)
classes = sorted(idx.species.unique())
y, sites = idx.species.to_numpy(), idx.location_id.astype(str).to_numpy()
has_a = idx.has_audio.astype(bool).to_numpy()
S = np.flatnonzero(idx.split == "source")
T = np.flatnonzero(idx.split.isin(["adapt", "test"]))
proj = idx.location_id.map(project).to_numpy()
print(f"clips {len(idx)} | source {len(S)} | target {len(T)} | classes {len(classes)}")

P_cam, P_aud = np.full((len(idx), len(classes)), np.nan), np.full((len(idx), len(classes)), np.nan)
fold = np.full(len(idx), -1)

# Source: out-of-fold probabilities from 5 folds grouped by site
for k, (a, b) in enumerate(GroupKFold(5).split(S, groups=sites[S])):
    P_cam[S[b]], P_aud[S[b]] = experts(S[a], S[b])
    fold[S[b]] = k
    print(f"source fold {k + 1}/5 done ({len(b)} clips)", flush=True)

# Target: model trained on all source clips
P_cam[T], P_aud[T] = experts(S, T)
print("target done")

# Check: target probabilities must match baselines.py
z = np.load(OUT / "predictions.npz", allow_pickle=True)
order = pd.Series(range(len(idx)), index=idx.video_name)[z["video_name"]].to_numpy()
assert list(z["classes"]) == classes, "class order differs from baselines.py"
print(f"check vs baselines.py: camera max |diff| {np.abs(P_cam[order] - z['global_visual']).max():.2e} | "
      f"audio max |diff| {np.abs(P_aud[order] - z['audio_only']).max():.2e} (both should be ~0)")

extra = {}
if args.with_l2:
    L2c, L2a = np.full_like(P_cam, np.nan), np.full_like(P_aud, np.nan)
    for p in np.unique(proj[S]):
        te, tr = S[proj[S] == p], S[proj[S] != p]
        n_tr = pd.Series(y[tr]).value_counts()
        te = te[n_tr.reindex(y[te]).fillna(0).to_numpy() >= MIN_TRAIN_L2]
        if len(te):
            L2c[te], L2a[te] = experts(tr, te)
        print(f"L2 held-out project {p}: {len(te)} scorable clips", flush=True)
    extra = {"camera_l2": L2c[S], "audio_l2": L2a[S], "video_name_l2": idx.video_name.to_numpy()[S]}

# Per-clip table: selector-safe columns plus analysis-only columns (ana_*)
tm = pd.read_csv(OUT / "test_meta.csv")
sm = pd.read_csv(OUT / "source_meta.csv")[["video_name", "is_ir", "placement"]]
pr = pd.read_csv(OUT / "predictions.csv")[["video_name", "target_sound"]]
tax = tm.drop_duplicates("species").set_index("species")[["class", "order", "family"]]

ev = idx[["video_name", "split", "species", "location_id", "has_audio"]].copy()
ev["project"] = proj
ev = ev.join(tax, on="species")
cond = pd.concat([sm, tm[["video_name", "is_ir", "placement"]]]).drop_duplicates("video_name")
ev = ev.merge(cond, on="video_name", how="left")
ev["prob_source"] = np.where(ev.split == "source", "oof_site_cv", "full_source")
ev["cv_fold"] = fold
for name, P in [("camera", P_cam), ("audio", P_aud)]:
    ev[f"pred_{name}"] = np.array(classes)[P.argmax(1)]
    ev[f"conf_{name}"] = P.max(1)
    ev[f"correct_{name}"] = ev[f"pred_{name}"] == ev.species
ana = tm[["video_name", "visible_frac", "mean_area", "exits_frame"]].merge(pr, on="video_name", how="left")
ana.columns = ["video_name"] + [f"ana_{c}" for c in ana.columns[1:]]
ev = ev.merge(ana, on="video_name", how="left")
assert (ev.video_name.to_numpy() == idx.video_name.to_numpy()).all()

ev.to_csv(OUT / "clip_events.csv", index=False)
np.savez(OUT / "clip_probs.npz", classes=np.array(classes), video_name=idx.video_name.to_numpy(),
         split=idx.split.to_numpy(), camera=P_cam, audio=P_aud, **extra)

for part, m in [("source (out-of-fold)", ev.split == "source"), ("target", ev.split != "source")]:
    d = ev[m]
    print(f"{part:22s} n={len(d)} | camera acc {d.correct_camera.mean():.3f} conf {d.conf_camera.mean():.3f} | "
          f"audio acc {d.correct_audio.mean():.3f} conf {d.conf_audio.mean():.3f} | "
          f"IR {d.is_ir.astype(str).str.lower().eq('true').mean():.2f} | placement unknown {(d.placement == 'unknown').mean():.2f}")
print(f"missing conditions: is_ir {ev.is_ir.isna().sum()}, placement {ev.placement.isna().sum()}, taxon {ev['class'].isna().sum()}")
print(f"Saved {OUT / 'clip_events.csv'} and {OUT / 'clip_probs.npz'}")
