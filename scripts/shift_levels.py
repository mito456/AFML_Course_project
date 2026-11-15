# Q1: how much each sensor drops from seen site to new site to new project (source cross-validation)
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import camera_expert as cam
from config import OUT, RESULTS, SEED
from plot_utils import AUDIO, CAMERA, INK2, W, err, plt, save

# A species is scored at the new-project level only if it has >= 5 training clips in other projects
MIN_TRAIN = 5
N_BOOT = 1000
LEVELS = ["L0_same_sites", "L1_new_sites", "L2_new_project"]


# Project = letters before the number in location_id
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
    return accuracy_score(yt, p), f1_score(yt, p, labels=np.unique(yt), average="macro", zero_division=0)


# 95% interval by resampling whole sites
def boot(yt, preds, sites):
    uniq = np.unique(sites)
    rows = {s: np.flatnonzero(sites == s) for s in uniq}
    rng = np.random.default_rng(SEED)
    out = {k: [] for k in preds}
    for _ in range(N_BOOT):
        r = np.concatenate([rows[s] for s in rng.choice(uniq, len(uniq))])
        for k, p in preds.items():
            out[k].append(metrics(yt[r], p[r]))
    return {k: np.array(v) for k, v in out.items()}


idx = pd.read_csv(OUT / "feat_index.csv")
V, A = np.load(OUT / "visual_emb.npy"), np.nan_to_num(np.load(OUT / "audio_emb.npy"))
idx["project"] = idx.location_id.map(project)
classes = sorted(idx.species.unique())
y, sites, proj = idx.species.to_numpy(), idx.location_id.astype(str).to_numpy(), idx.project.to_numpy()
has_a = idx.has_audio.astype(bool).to_numpy()
S = np.flatnonzero(idx.split == "source")
T = np.flatnonzero(idx.split.isin(["adapt", "test"]))
ys, ss, ps = y[S], sites[S], proj[S]
frames = cam.load_frames(idx)
sensors = {"camera": np.ones(len(idx), bool), "audio": has_a}


def predict(sensor, tr, te):
    if sensor == "camera":
        return cam.fit_predict(frames, y[tr], tr, te, classes)
    return fit_predict(A[tr], y[tr], A[te])


print(f"source clips {len(S)} | sites {len(np.unique(ss))} | projects {len(np.unique(ps))} | "
      f"target clips {len(T)} | target projects also in source: "
      f"{len(set(proj[T]) & set(ps))} of {len(set(proj[T]))}")

# Seen site = random folds, new site = folds grouped by site, new project = leave one project out
folds = {
    "L0_same_sites": list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(ys, ys)),
    "L1_new_sites": list(GroupKFold(5).split(ys, ys, ss)),
    "L2_new_project": [(np.flatnonzero(ps != p), np.flatnonzero(ps == p)) for p in np.unique(ps)],
}
covered = np.zeros(len(S), bool)
for tr, te in folds["L2_new_project"]:
    n_tr = pd.Series(ys[tr]).value_counts()
    covered[te] = n_tr.reindex(ys[te]).fillna(0).to_numpy() >= MIN_TRAIN

print(f"\n=== L2 coverage: {covered.sum()} of {len(S)} source clips scorable "
      f"(species has >= {MIN_TRAIN} training clips in the other projects) ===")
cov = pd.DataFrame({"project": ps, "species": ys, "covered": covered})
lost = cov[~cov.covered].groupby(["project", "species"]).size()
print("not scorable (project, species: clips):", ", ".join(f"{p}/{s}: {n}" for (p, s), n in lost.items()) or "none")

# Out-of-fold predictions at each level, plus the official target with a model trained on all source clips
preds = {}
for name, ok in sensors.items():
    oks = ok[S]
    preds[name] = {}
    for level, fl in folds.items():
        p = np.empty(len(S), object)
        for tr, te in fl:
            tr = tr[oks[tr]]
            p[te] = predict(name, S[tr], S[te])
        p[~oks] = classes[0]
        preds[name][level] = p
    tr = S[ok[S]]
    pt = predict(name, tr, T)
    pt[~ok[T]] = classes[0]
    preds[name]["target_official"] = pt

rows = []


def add(sensor, level, clip_set, yt, st, draws, point):
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    for j, metric in enumerate(["accuracy", "macro_f1"]):
        rows.append({"sensor": sensor, "level": level, "clip_set": clip_set, "n_clips": len(yt),
                     "n_sites": len(np.unique(st)), "metric": metric,
                     "value": point[j], "ci_lo": lo[j], "ci_hi": hi[j]})


print("\n=== Each level on its own scored clips (site-bootstrap 95% CI) ===")
for sensor in sensors:
    for level in LEVELS + ["target_official"]:
        if level == "target_official":
            yt, st, p, cs = y[T], sites[T], preds[sensor][level], "target"
        else:
            m = covered if level == "L2_new_project" else np.ones(len(S), bool)
            yt, st, p, cs = ys[m], ss[m], preds[sensor][level][m], "own"
        d = boot(yt, {level: p}, st)[level]
        add(sensor, level, cs, yt, st, d, metrics(yt, p))
        r = rows[-2:]
        print(f"{sensor:6s} {level:16s} n={len(yt):4d} sites={len(np.unique(st)):3d} | "
              f"acc {r[0]['value']:.3f} [{r[0]['ci_lo']:.3f}, {r[0]['ci_hi']:.3f}]  "
              f"macro-F1 {r[1]['value']:.3f} [{r[1]['ci_lo']:.3f}, {r[1]['ci_hi']:.3f}]")

# Paired drops on the clips scored at every level
print(f"\n=== Paired drops on the common clip set (the {covered.sum()} clips scorable at every level) ===")
yc, sc = ys[covered], ss[covered]
for sensor in sensors:
    pc = {lvl: preds[sensor][lvl][covered] for lvl in LEVELS}
    d = boot(yc, pc, sc)
    for lvl in LEVELS:
        add(sensor, lvl, "common", yc, sc, d[lvl], metrics(yc, pc[lvl]))
    for a, b in [("L0_same_sites", "L1_new_sites"), ("L1_new_sites", "L2_new_project"),
                 ("L0_same_sites", "L2_new_project")]:
        point = np.subtract(metrics(yc, pc[a]), metrics(yc, pc[b]))
        add(sensor, f"drop {a[:2]}->{b[:2]}", "common", yc, sc, d[a] - d[b], point)
        r = rows[-2:]
        print(f"{sensor:6s} drop {a[:2]}->{b[:2]} | acc {r[0]['value']:+.3f} [{r[0]['ci_lo']:+.3f}, {r[0]['ci_hi']:+.3f}]  "
              f"macro-F1 {r[1]['value']:+.3f} [{r[1]['ci_lo']:+.3f}, {r[1]['ci_hi']:+.3f}]")

print("\n=== L2 per held-out project (accuracy on scorable clips; small projects are noise) ===")
for p in np.unique(ps):
    m = (ps == p) & covered
    if m.sum() == 0:
        print(f"{p:9s} scorable 0 of {(ps == p).sum()}")
        continue
    accs = {s: accuracy_score(ys[m], preds[s]["L2_new_project"][m]) for s in sensors}
    print(f"{p:9s} scorable {m.sum():3d} of {(ps == p).sum():3d} | camera {accs['camera']:.3f}  audio {accs['audio']:.3f}")

pd.DataFrame(rows).to_csv(RESULTS / "shift_results.csv", index=False)
# Out-of-fold predictions saved for shift_drop_diff.py
oof = idx.iloc[S][["video_name", "species", "location_id", "project"]].reset_index(drop=True)
oof["covered_L2"] = covered
for sensor in sensors:
    for lvl in LEVELS:
        oof[f"pred_{sensor}_{lvl}"] = preds[sensor][lvl]
oof.to_csv(OUT / "shift_oof.csv", index=False)
print(f"\nSaved {RESULTS / 'shift_results.csv'} and {OUT / 'shift_oof.csv'}")

# Fig 1: macro-F1 at each shift level on the common clip set
s = pd.read_csv(RESULTS / "shift_results.csv")
names = ["Seen site", "New site", "New project"]
fig, ax = plt.subplots(figsize=(W, 2.6))
for sensor, col, dx in [("camera", CAMERA, -0.04), ("audio", AUDIO, 0.04)]:
    d = s[(s.sensor == sensor) & (s.metric == "macro_f1") & s.level.isin(LEVELS) & (s.clip_set == "common")]
    d = d.set_index("level").loc[LEVELS]
    x = np.arange(3) + dx
    ax.errorbar(x, 100 * d.value, yerr=100 * err(d.value, d.ci_lo, d.ci_hi), color=col, lw=2, marker="o", ms=5,
                capsize=3, label=sensor.capitalize())
    for xi, v in zip(x, d.value):
        ax.annotate(f"{100 * v:.1f}", (xi, 100 * v), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK2)
ax.set_xticks(range(3), names)
ax.set_ylabel("Macro-F1 (%)")
ax.set_ylim(0, 72)
ax.legend(loc="upper right")
save(fig, "fig1_shift_levels")
