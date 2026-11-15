# Few-label adaptation: retrain both experts on source + k labelled target clips per species (k = 1, 3; seeds 0-4)
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import camera_expert as cam
from config import OUT, RESULTS, SEED
from plot_utils import AUDIO, CAMERA, GREY, INK2, W as FIG_W, err, plt, save

SENSORS = ["camera", "audio"]
# Candidate weights for labelled target clips, chosen on source pseudo-targets
W_GRID = [1.0, 5.0, 20.0]
K_TUNE = 3
TIE = 0.005
MIN_HELDOUT = 2
MAX_TRIES = 10000
MARGIN = 0.03
N_BOOT = 1000
# Unlabelled clip groups scored: all, held-out sites, pool sites, pool cameras with / without a label of that species
SETS = ["all_unlabelled", "heldout_sites", "pool_sites", "pool_same_species", "pool_other_species"]
TUNE_FILE = RESULTS / "adapt_weight_tuning.csv"


def project(loc):
    l = str(loc)
    if l == "nan":
        return "unknown"
    if re.fullmatch(r"[0-9a-f]{8}", l):
        return "hash_site"
    m = re.match(r"[A-Za-z]+", l)
    p = m.group(0).upper() if m else "unknown"
    return "COLPA" if p == "COPLA" else p


# Audio expert with clip weights; clips without audio are left out of training and get uniform probabilities
def audio_proba(tr, te, w_tr):
    keep = has_a[tr]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    m.fit(A[tr[keep]], y[tr[keep]], logisticregression__sample_weight=w_tr[keep])
    p = np.zeros((len(te), len(classes)))
    p[:, [classes.index(c) for c in m.classes_]] = m.predict_proba(A[te])
    p[~has_a[te]] = 1.0 / len(classes)
    return p


# Trains both experts on source plus labelled clips (labelled clips get weight w)
def experts(base, lab, w, te, sensors=SENSORS):
    tr = np.concatenate([base, lab]).astype(int)
    out = {}
    for sen in sensors:
        wt = np.concatenate([np.ones(len(base)), np.full(len(lab), w[sen])])
        out[sen] = (cam.fit_proba(frames, y[tr], tr, te, classes, w_tr=wt) if sen == "camera"
                    else audio_proba(tr, te, wt))
    return out


def metrics(yi, pred, conf, with_ece=False):
    hit = pred == yi
    n = len(classes)
    tp = np.bincount(yi[hit], minlength=n)
    n_true, n_pred = np.bincount(yi, minlength=n), np.bincount(pred, minlength=n)
    f1 = 2 * tp / np.maximum(n_true + n_pred, 1)
    out = [hit.mean(), f1[n_true > 0].mean(), conf.mean() - hit.mean()]
    if with_ece:
        b = np.minimum((conf * 10).astype(int), 9)
        out.append(sum(abs(hit[b == j].mean() - conf[b == j].mean()) * (b == j).mean()
                       for j in range(10) if (b == j).any()))
    return np.array(out)


# Splits target sites per project into a pool (can be labelled) and held-out sites (never labelled)
def site_split(loc, site_proj, sp, rng, k_need, min_heldout, max_tries):
    by_proj = {p: np.array(g.index) for p, g in site_proj.groupby(site_proj)}
    species = np.unique(sp)
    for tries in range(1, max_tries + 1):
        pool = []
        for s in by_proj.values():
            s = rng.permutation(s)
            pool += list(s[:len(s) // 2 + int(len(s) % 2 == 1 and rng.random() < 0.5)])
        ip = np.isin(loc, pool)
        cp = pd.Series(sp[ip]).value_counts().reindex(species).fillna(0)
        ch = pd.Series(sp[~ip]).value_counts().reindex(species).fillna(0)
        if (cp >= k_need).all() and (ch >= min_heldout).all():
            return ip, tries
    raise RuntimeError(f"no valid site split in {max_tries} tries")


# Orders pool clips so labels are spread over cameras
def spread_rank(loc, sp, ip, rng):
    d = pd.DataFrame({"loc": loc, "sp": sp, "u": rng.random(len(loc)), "v": rng.random(len(loc))})[ip]
    d["round"] = d.sort_values("u").groupby(["sp", "loc"]).cumcount()
    d = d.sort_values(["sp", "round", "v"])
    rank = np.full(len(loc), 10 ** 9)
    rank[d.index.to_numpy()] = d.groupby("sp").cumcount().to_numpy()
    return rank


def clip_sets(loc, sp, ip, rank, k):
    lab = ip & (rank < k)
    pairs = set(zip(loc[lab], sp[lab]))
    same = np.array([(l, s) in pairs for l, s in zip(loc, sp)])
    return lab, {"all_unlabelled": ~lab, "heldout_sites": ~ip, "pool_sites": ip & ~lab,
                 "pool_same_species": ip & ~lab & same, "pool_other_species": ip & ~lab & ~same}


# --phase tune picks the weight on source; --phase target runs the adaptation on the target
ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=["tune", "target"], required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--ks", type=int, nargs="+", default=[1, 3])
ap.add_argument("--w-camera", type=float, default=None)
ap.add_argument("--w-audio", type=float, default=None)
args = ap.parse_args()

idx = pd.read_csv(OUT / "feat_index.csv")
A = np.nan_to_num(np.load(OUT / "audio_emb.npy"))
frames = cam.load_frames(idx)
classes = sorted(idx.species.unique())
y, sites = idx.species.to_numpy(), idx.location_id.astype(str).to_numpy()
yi_all = pd.Series(range(len(classes)), index=classes)[y].to_numpy()
has_a = idx.has_audio.astype(bool).to_numpy()
proj = idx.location_id.map(project).to_numpy()
S = np.flatnonzero(idx.split == "source")
T = np.flatnonzero(idx.split.isin(["adapt", "test"]))
print(f"source {len(S)} | target {len(T)} clips, {len(np.unique(sites[T]))} sites | classes {len(classes)}")

if args.phase == "tune":
    # Tuning: 3 site-grouped source folds act as pseudo-targets; picks the smallest weight within TIE of the best
    rows, rng = [], np.random.default_rng(SEED)
    for f, (a, b) in enumerate(GroupKFold(3).split(S, groups=sites[S])):
        base, pseudo = S[a], S[b]
        loc, sp, yi = sites[pseudo], y[pseudo], yi_all[pseudo]
        site_proj = pd.Series(proj[pseudo], index=loc).groupby(level=0).first()
        ip, _ = site_split(loc, site_proj, sp, rng, 0, 0, 1)
        rank = spread_rank(loc, sp, ip, rng)
        lab, m = clip_sets(loc, sp, ip, rank, K_TUNE)
        print(f"fold {f + 1}/3: train {len(base)} source clips | pseudo-target {len(pseudo)} clips, "
              f"{len(site_proj)} sites | labels {lab.sum()} | held-out {m['heldout_sites'].sum()} | "
              f"pool unlabelled {m['pool_sites'].sum()}", flush=True)
        for w in [0.0] + W_GRID:
            P = experts(base, pseudo[lab] if w > 0 else pseudo[:0], {s: w for s in SENSORS}, pseudo)
            for sen, p in P.items():
                pred = p.argmax(1)
                for name in ["all_unlabelled", "heldout_sites", "pool_sites"]:
                    r = m[name]
                    rows.append({"fold": f, "sensor": sen, "w": w, "clip_set": name, "n_clips": int(r.sum()),
                                 "n_labelled": int(lab.sum()), "accuracy": (pred[r] == yi[r]).mean()})
            print(f"  w={w:g} done", flush=True)
    t = pd.DataFrame(rows)
    ref = t[t.w == 0].set_index(["fold", "sensor", "clip_set"]).accuracy
    t["gain"] = t.accuracy.to_numpy() - ref.loc[list(zip(t.fold, t.sensor, t.clip_set))].to_numpy()
    summ = t[t.w > 0].groupby(["sensor", "clip_set", "w"]).gain.mean().unstack("clip_set")
    chosen = {}
    for sen in SENSORS:
        g = summ.loc[sen, "all_unlabelled"]
        chosen[sen] = float(min(w for w, v in g.items() if v >= g.max() - TIE))
    t["chosen"] = [w == chosen[s] for s, w in zip(t.sensor, t.w)]
    t.to_csv(TUNE_FILE, index=False)
    print("\n=== Mean accuracy gain over w = 0 on pseudo-targets (3 source folds) ===")
    print((summ * 100).round(1).to_string())
    print(f"chosen w: {chosen}  -> saved {TUNE_FILE}")
    raise SystemExit

# Weights from the tuning file unless given on the command line
if args.w_camera is not None and args.w_audio is not None:
    W = {"camera": args.w_camera, "audio": args.w_audio}
else:
    tt = pd.read_csv(TUNE_FILE)
    W = tt[tt.chosen].groupby("sensor").w.first().to_dict()
    W.update({k: v for k, v in [("camera", args.w_camera), ("audio", args.w_audio)] if v is not None})
print(f"weights for labelled target clips: {W}")

# Global (unadapted) probabilities to compare against
z = np.load(OUT / "clip_probs.npz", allow_pickle=True)
assert (z["video_name"] == idx.video_name.to_numpy()).all() and list(z["classes"]) == classes, \
    "clip_probs.npz does not match feat_index.csv -> re-run global_expert_probs.py"
G = {sen: z[sen][T] for sen in SENSORS}
loc_t, sp_t, yt = sites[T], y[T], yi_all[T]
site_proj_t = pd.Series(proj[T], index=loc_t).groupby(level=0).first()
gpc = {sen: (G[sen].argmax(1), G[sen].max(1)) for sen in SENSORS}
print(f"check global on all target: camera acc {metrics(yt, *gpc['camera'])[0]:.3f} (baselines 0.524) | "
      f"audio acc {metrics(yt, *gpc['audio'])[0]:.3f} (baselines 0.170)")

probs, M, draw_rows, split_rows = {}, {}, [], []
# Per seed: split sites, draw k labels per species, refit, store probabilities
for s in args.seeds:
    rng = np.random.default_rng(s)
    ip, tries = site_split(loc_t, site_proj_t, sp_t, rng, max(args.ks), MIN_HELDOUT, MAX_TRIES)
    rank = spread_rank(loc_t, sp_t, ip, rng)
    pool_sites = set(loc_t[ip])
    split_rows += [{"seed": s, "location_id": l, "project": p, "role": "pool" if l in pool_sites else "heldout"}
                   for l, p in site_proj_t.items()]
    for k in args.ks:
        lab, m = clip_sets(loc_t, sp_t, ip, rank, k)
        assert (pd.Series(sp_t[lab]).value_counts() == k).all() and lab.sum() == k * len(classes)
        M[(k, s)] = m
        draw_rows += [{"seed": s, "k": k, "video_name": v, "species": sp, "location_id": l}
                      for v, sp, l in zip(idx.video_name.to_numpy()[T][lab], sp_t[lab], loc_t[lab])]
        P = experts(S, T[lab], W, T)
        for sen in SENSORS:
            probs[(sen, k, s)] = P[sen]
        pc, r = P["camera"].argmax(1), m["all_unlabelled"]
        print(f"seed {s} k={k} (split after {tries} tries): {len(pool_sites)} pool / "
              f"{len(site_proj_t) - len(pool_sites)} held-out sites | labels {lab.sum()} at "
              f"{len(set(loc_t[lab]))} sites | " + ", ".join(f"{n} {m[n].sum()}" for n in SETS)
              + f" | camera acc adapted {(pc[r] == yt[r]).mean():.3f} vs global {(gpc['camera'][0][r] == yt[r]).mean():.3f}",
              flush=True)

pd.DataFrame(split_rows).to_csv(OUT / "adapt_site_split.csv", index=False)
pd.DataFrame(draw_rows).to_csv(OUT / "adapt_draws.csv", index=False)
np.savez(OUT / "adapted_probs.npz", classes=np.array(classes), video_name=idx.video_name.to_numpy()[T],
         w_camera=W["camera"], w_audio=W["audio"],
         **{f"{sen}_k{k}_s{s}": p for (sen, k, s), p in probs.items()},
         **{f"eval_k{k}_s{s}": m["all_unlabelled"] for (k, s), m in M.items()},
         **{f"eval_heldout_k{k}_s{s}": m["heldout_sites"] for (k, s), m in M.items()},
         **{f"eval_pool_k{k}_s{s}": m["pool_sites"] for (k, s), m in M.items()})

apc = {key: (p.argmax(1), p.max(1)) for key, p in probs.items()}


# Adapted vs global metrics and selection headroom (oracle - camera) per clip group, averaged over seeds
def table(rows):
    out = {}
    for k in args.ks:
        for name in SETS:
            per, head = {sen: [] for sen in SENSORS}, []
            for s in args.seeds:
                r = rows[M[(k, s)][name][rows]]
                if len(r) == 0:
                    for sen in SENSORS:
                        per[sen].append(np.full((2, 3), np.nan))
                    head.append([np.nan, np.nan])
                    continue
                for sen in SENSORS:
                    per[sen].append([metrics(yt[r], apc[(sen, k, s)][0][r], apc[(sen, k, s)][1][r]),
                                     metrics(yt[r], gpc[sen][0][r], gpc[sen][1][r])])
                hits = {(sen, src): (pr[r] == yt[r]) for sen in SENSORS
                        for src, pr in [("a", apc[(sen, k, s)][0]), ("g", gpc[sen][0])]}
                head.append([(hits[("camera", c)] | hits[("audio", c)]).mean() - hits[("camera", c)].mean()
                             for c in ["a", "g"]])
            for sen in SENSORS:
                out[(sen, k, name)] = np.nanmean(per[sen], axis=0)
            out[("headroom", k, name)] = np.nanmean(head, axis=0)
    return out


point = table(np.arange(len(T)))
# Site-bootstrap 95% intervals
uniq = np.unique(loc_t)
members = {u: np.flatnonzero(loc_t == u) for u in uniq}
rng = np.random.default_rng(SEED)
draws = {key: [] for key in point}
for b in range(N_BOOT):
    rr = np.concatenate([members[u] for u in rng.choice(uniq, len(uniq))])
    for key, v in table(rr).items():
        draws[key].append(v)
    if (b + 1) % 200 == 0:
        print(f"bootstrap {b + 1}/{N_BOOT}", flush=True)
D = {key: np.array(v) for key, v in draws.items()}
lo = lambda x: np.nanpercentile(x, 2.5, axis=0)
hi = lambda x: np.nanpercentile(x, 97.5, axis=0)

res, MET = [], ["accuracy", "macro_f1", "overconfidence"]
for sen in SENSORS:
    for k in args.ks:
        for name in SETS:
            v, d = point[(sen, k, name)], D[(sen, k, name)]
            per_seed = []
            for s in args.seeds:
                r = M[(k, s)][name]
                per_seed.append(metrics(yt[r], apc[(sen, k, s)][0][r], apc[(sen, k, s)][1][r], with_ece=True)
                                if r.any() else np.full(4, np.nan))
            ece_g = np.nanmean([metrics(yt[M[(k, s)][name]], gpc[sen][0][M[(k, s)][name]],
                                        gpc[sen][1][M[(k, s)][name]], with_ece=True)[3]
                                for s in args.seeds if M[(k, s)][name].any()])
            common = {"sensor": sen, "k": k, "clip_set": name,
                      "n_clips_mean": np.mean([M[(k, s)][name].sum() for s in args.seeds]),
                      "n_species_mean": np.mean([len(np.unique(yt[M[(k, s)][name]])) for s in args.seeds])}
            for j, met in enumerate(MET):
                diff = d[:, 0, j] - d[:, 1, j]
                res.append({**common, "metric": met, "adapted": v[0, j], "adapted_lo": lo(d[:, 0, j]),
                            "adapted_hi": hi(d[:, 0, j]), "global": v[1, j], "diff": v[0, j] - v[1, j],
                            "diff_lo": lo(diff), "diff_hi": hi(diff),
                            "adapted_seed_min": np.nanmin([p[j] for p in per_seed]),
                            "adapted_seed_max": np.nanmax([p[j] for p in per_seed])})
            e = np.nanmean([p[3] for p in per_seed])
            res.append({**common, "metric": "ece_point_only", "adapted": e, "global": ece_g, "diff": e - ece_g})

# Pool - held-out and same - other species gaps
for sen in SENSORS:
    for k in args.ks:
        for c_name, (a_set, b_set) in {"pool_minus_heldout": ("pool_sites", "heldout_sites"),
                                       "same_minus_other_species": ("pool_same_species", "pool_other_species")}.items():
            for j, met in enumerate(MET[:2]):
                gain = lambda arr, n: arr[(sen, k, n)][..., 0, j] - arr[(sen, k, n)][..., 1, j]
                gap = gain(D, a_set) - gain(D, b_set)
                res.append({"sensor": sen, "k": k, "clip_set": c_name, "metric": met,
                            "diff": gain(point, a_set) - gain(point, b_set), "diff_lo": lo(gap), "diff_hi": hi(gap)})

# Selection headroom before and after adapting
for k in args.ks:
    for name in ["all_unlabelled", "heldout_sites", "pool_sites"]:
        v, d = point[("headroom", k, name)], D[("headroom", k, name)]
        res.append({"sensor": "oracle_headroom", "k": k, "clip_set": name, "metric": "accuracy",
                    "adapted": v[0], "adapted_lo": lo(d[:, 0]), "adapted_hi": hi(d[:, 0]), "global": v[1],
                    "diff": v[0] - v[1], "diff_lo": lo(d[:, 0] - d[:, 1]), "diff_hi": hi(d[:, 0] - d[:, 1])})

res = pd.DataFrame(res)
res.to_csv(RESULTS / "adapted_results.csv", index=False)
pd.set_option("display.width", 220)
cols = ["sensor", "k", "clip_set", "metric", "n_clips_mean", "adapted", "global", "diff", "diff_lo", "diff_hi"]
print("\n=== Adapted vs global (seed-averaged, 95% site CI; diff = adapted - global) ===")
print(res[res.metric.isin(["accuracy", "macro_f1"])][cols].round(3).to_string(index=False))
print("\n=== Over-confidence (mean conf - acc) and ECE (point only) ===")
print(res[res.metric.isin(["overconfidence", "ece_point_only"])][cols].round(3).to_string(index=False))


def row(sen, k, cs, met="accuracy"):
    return res[(res.sensor == sen) & (res.k == k) & (res.clip_set == cs) & (res.metric == met)].iloc[0]


# Pre-registered hypotheses H1-H6
if 3 in args.ks:
    fmt = lambda r: f"{100 * r['diff']:+.1f} [{100 * r.diff_lo:+.1f}, {100 * r.diff_hi:+.1f}]"
    h1, h2 = row("camera", 3, "pool_sites"), row("camera", 3, "heldout_sites")
    h3, h4 = row("camera", 3, "pool_minus_heldout"), row("camera", 3, "same_minus_other_species")
    h2v = ("supported: no useful region transfer" if h2.diff_hi < MARGIN else
           "rejected: region transfer exists" if h2.diff_lo > 0 else "inconclusive")
    print("\n=== Hypotheses (accuracy points) ===")
    print(f"H1 pool-site gain {fmt(h1)} -> {'supported' if h1.diff_lo > 0 else 'not supported'}")
    print(f"H2 held-out gain {fmt(h2)} (margin +{100 * MARGIN:.0f}) -> {h2v}")
    print(f"H3 pool - held-out gain {fmt(h3)} -> {'supported' if h3.diff_lo > 0 else 'not supported'}")
    print(f"H4 same - other species (exploratory) {fmt(h4)}")
    print(f"H5 audio: pool {fmt(row('audio', 3, 'pool_sites'))}, held-out {fmt(row('audio', 3, 'heldout_sites'))}, "
          f"difference {fmt(row('audio', 3, 'pool_minus_heldout'))}")
    hr = row("oracle_headroom", 3, "all_unlabelled")
    print(f"H6 selection headroom: adapted {100 * hr.adapted:.1f} vs global {100 * hr['global']:.1f} -> "
          f"change {fmt(hr)}")
print(f"\nSaved adapt_site_split.csv, adapt_draws.csv, adapted_probs.npz (outputs/) and "
      f"{RESULTS / 'adapted_results.csv'}")

if 3 in args.ks:
    # Fig 2: adaptation gain by clip group (k = 3)
    a = res
    sets = ["heldout_sites", "pool_same_species", "pool_other_species", "all_unlabelled"]
    labels = ["New cameras\n(never\nlabelled)", "Labelled\ncameras,\nsame species", "Labelled\ncameras,\nother species",
              "All\nunlabelled\nclips"]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.9))
    x = np.arange(len(sets))
    lims = []
    for sensor, col, dx in [("camera", CAMERA, -0.19), ("audio", AUDIO, 0.19)]:
        d = a[(a.sensor == sensor) & (a.k == 3) & (a.metric == "accuracy")].set_index("clip_set").loc[sets]
        v = 100 * d["diff"]
        ax.bar(x + dx, v, width=0.36, color=col, label=f"{sensor.capitalize()} (weight x{W[sensor]:g})",
               edgecolor="white", linewidth=1)
        ax.errorbar(x + dx, v, yerr=100 * err(d["diff"], d.diff_lo, d.diff_hi), fmt="none", ecolor=INK2, capsize=2.5,
                    lw=0.9)
        for xi, vi, hi_, lo_ in zip(x + dx, v, 100 * d.diff_hi, 100 * d.diff_lo):
            ax.annotate(f"{vi:+.1f}", (xi, hi_ if vi >= 0 else lo_), xytext=(0, 3 if vi >= 0 else -9),
                        textcoords="offset points", ha="center", fontsize=7, color=INK2)
        lims += [100 * d.diff_lo.min(), 100 * d.diff_hi.max()]
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(min(lims) - 2.5, max(lims) + 2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy gain vs global (points)")
    ax.legend(loc="upper right")
    save(fig, "fig2_adapt_decomp")

    # Fig 3: selection headroom before and after adapting (k = 3)
    sets = ["heldout_sites", "pool_sites", "all_unlabelled"]
    labels = ["New cameras", "Labelled cameras\n(unlabelled clips)", "All unlabelled clips"]
    d = a[(a.sensor == "oracle_headroom") & (a.k == 3)].set_index("clip_set").loc[sets]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.5))
    x = np.arange(len(sets))
    for lab, vals, col, dx in [("Global experts", d["global"], GREY, -0.19), ("Adapted experts (k = 3)", d.adapted, CAMERA, 0.19)]:
        ax.bar(x + dx, 100 * vals, width=0.36, color=col, label=lab, edgecolor="white", linewidth=1)
        for xi, v in zip(x + dx, 100 * vals):
            ax.annotate(f"{v:.1f}", (xi, v / 2), ha="center", va="center", fontsize=7, color="white")
    ax.errorbar(x + 0.19, 100 * d.adapted, yerr=100 * err(d.adapted, d.adapted_lo, d.adapted_hi), fmt="none",
                ecolor=INK2, capsize=2.5, lw=0.9)
    for xi, (dv, lo_, hi_) in zip(x, zip(d["diff"], d.diff_lo, d.diff_hi)):
        ax.annotate(f"change {100 * dv:+.1f}\n[{100 * lo_:+.1f}, {100 * hi_:+.1f}]", (xi, 11.4), ha="center", fontsize=6.5,
                    color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Oracle minus camera (accuracy points)")
    ax.set_ylim(0, 13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, "fig3_headroom")
