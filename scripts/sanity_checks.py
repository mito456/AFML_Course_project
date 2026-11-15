# Sanity checks: alignment, reproducing baselines.py, chance levels, shuffled labels, site-bootstrap intervals, confusions
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import camera_expert as cam
from config import OUT, SEED

rng = np.random.default_rng(SEED)

split = pd.read_csv(OUT / "split.csv")
idx = pd.read_csv(OUT / "feat_index.csv")
V = np.load(OUT / "visual_emb.npy")
A = np.load(OUT / "audio_emb.npy")
print("split columns:", split.columns.tolist())
print("feat_index columns:", idx.columns.tolist())

print("\n=== CHECK 1: alignment ===")
assert len(idx) == len(V) == len(A), f"row counts differ: {len(idx)}, {len(V)}, {len(A)}"
print("rows in feature files:", len(idx))
print("duplicate video names in feat_index:", idx["video_name"].duplicated().sum())
df = idx.merge(split, on="video_name", how="left", suffixes=("", "_split"))
print("feature rows with no label in split.csv:", df["species"].isna().sum())
if "species_split" in df.columns:
    print("species mismatches between files:", (df["species"] != df["species_split"]).sum())
print("\nSpot-check 5 random rows (open these videos and confirm the species):")
print(df.sample(5, random_state=SEED)[["video_name", "species", "location_id", "split"]])

train = (df["split"] == "source").values
target = df["split"].isin(["adapt", "test"]).values
classes = sorted(df.loc[train, "species"].unique())
y = df["species"].values
sites = df["location_id"].astype(str).values
no_audio = ~df["has_audio"].astype(bool).values
A = np.nan_to_num(A)
train_a = train & ~no_audio
print(f"\ntrain clips: {train.sum()} (with audio: {train_a.sum()}), target clips: {target.sum()}, "
      f"target clips without audio: {(no_audio & target).sum()}")

# Audio expert, same as baselines.py
def fit_predict(X, y_train_override=None, fit_mask=None):
    fit_mask = train if fit_mask is None else fit_mask
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, class_weight="balanced"))
    ytr = y[fit_mask] if y_train_override is None else y_train_override
    model.fit(X[fit_mask], ytr)
    return model.predict(X[target])

frames = cam.load_frames(idx)
tr_i, te_i = np.flatnonzero(train), np.flatnonzero(target)


def fit_predict_camera(y_train=None):
    return cam.fit_predict(frames, y[train] if y_train is None else y_train, tr_i, te_i, classes)

def score(pred, mask=None):
    yt = y[target]
    if mask is not None:
        yt, pred = yt[mask], pred[mask]
    return accuracy_score(yt, pred), f1_score(yt, pred, labels=classes, average="macro", zero_division=0)

print("\n=== CHECK 2: reproduce baseline numbers ===")
pred_v = fit_predict_camera()
pred_a = fit_predict(A, fit_mask=train_a)
na_t = no_audio[target]
pred_a[na_t] = classes[0]
print(f"no-audio target clips are predicted as '{classes[0]}' (as in baselines.py); "
      f"{(y[target][na_t] == classes[0]).sum()} of {na_t.sum()} are truly '{classes[0]}'")
for name, p in [("camera", pred_v), ("audio", pred_a)]:
    acc, f1 = score(p)
    print(f"{name:7s} accuracy {acc:.3f}  macro-F1 {f1:.3f}")
acc, f1 = score(pred_a, ~na_t)
print(f"audio (clips with audio only) accuracy {acc:.3f}  macro-F1 {f1:.3f}")

print("\n=== CHECK 3: chance baselines ===")
counts = pd.Series(y[target]).value_counts(normalize=True)
print(f"uniform guess: {1/len(classes):.3f}")
print(f"always '{counts.index[0]}' (most common in target): {counts.iloc[0]:.3f}")

print("\n=== CHECK 4: shuffled training labels (should fall to chance) ===")
for name, X, m in [("camera", V, train), ("audio", A, train_a)]:
    accs = []
    for s in range(5):
        shuffled = np.random.default_rng(s).permutation(y[m])
        p = fit_predict_camera(shuffled) if name == "camera" else fit_predict(X, shuffled, fit_mask=m)
        if name == "audio":
            p[na_t] = classes[0]
        accs.append(accuracy_score(y[target], p))
    print(f"{name:7s} shuffled accuracy: mean {np.mean(accs):.3f}  (runs: {np.round(accs, 3)})")

print("\n=== CHECK 5: 95% CIs, resampling whole sites (1000 draws) ===")
t_sites = sites[target]
uniq = np.unique(t_sites)
site_rows = {s: np.where(t_sites == s)[0] for s in uniq}
yt = y[target]

def boot(pred):
    accs, f1s = [], []
    for _ in range(1000):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([site_rows[s] for s in draw])
        accs.append(accuracy_score(yt[rows], pred[rows]))
        f1s.append(f1_score(yt[rows], pred[rows], average="macro"))
    return np.percentile(accs, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5])

for name, p in [("camera", pred_v), ("audio", pred_a)]:
    (a_lo, a_hi), (f_lo, f_hi) = boot(p)
    print(f"{name:7s} accuracy [{a_lo:.3f}, {a_hi:.3f}]  macro-F1 [{f_lo:.3f}, {f_hi:.3f}]")
print(f"target sites: {len(uniq)}, median clips per site: "
      f"{np.median([len(r) for r in site_rows.values()]):.0f}")

print("\n=== CHECK 6: top 8 confusions (true -> predicted : count) ===")
for name, p in [("camera", pred_v), ("audio", pred_a)]:
    cm = pd.DataFrame(confusion_matrix(yt, p, labels=classes), index=classes, columns=classes)
    pairs = cm.stack()
    pairs = pairs[[a != b for a, b in pairs.index]].sort_values(ascending=False).head(8)
    print(f"\n{name}:")
    for (t, pr), c in pairs.items():
        print(f"  {t} -> {pr} : {c}")
