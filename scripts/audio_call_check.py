# Checks whether audio is more accurate on clips with a clear animal sound (AudioSet call score)
import numpy as np
import pandas as pd

from config import OUT, RESULTS, SEED
from plot_utils import AUDIO, CAMERA, INK2, W, plt, save

N_BOOT = 1000
TOP_SHARE = 0.25
CLEAR = 0.3
MIN_PER_GROUP = 3

# AudioSet labels counted as animal vocalisations / movement sounds
VOCAL = ["Bird vocalization, bird call, bird song", "Chirp, tweet", "Squawk", "Fowl", "Chicken, rooster", "Cluck",
         "Turkey", "Gobble", "Coo", "Pigeon, dove", "Crow", "Caw", "Hoot", "Bird",
         "Grunt", "Snort", "Whimper", "Chatter", "Growling", "Roar", "Roaring cats (lions, tigers)", "Pig",
         "Oink", "Squeal", "Hiss", "Purr", "Bark", "Howl", "Squeak", "Rodents, rats, mice"]
MOVEMENT = ["Walk, footsteps", "Rustle", "Scratch", "Chewing, mastication", "Bird flight, flapping wings",
            "Wild animals"]

ev = pd.read_csv(OUT / "clip_events.csv")
idx = pd.read_csv(OUT / "feat_index.csv")
assert (ev.video_name.values == idx.video_name.values).all(), "clip_events and feat_index rows differ"
P = np.load(OUT / "audioset_probs.npy")
labels = pd.read_csv(OUT / "audioset_labels.csv").label.tolist()
col = {l: i for i, l in enumerate(labels)}
missing = [n for n in VOCAL + MOVEMENT if n not in col]
if missing:
    print(f"AudioSet labels not found (skipped): {missing}")
score = lambda names: P[:, [col[n] for n in names if n in col]].max(1)
ev["call_score"] = score(VOCAL)
ev["animal_score"] = score(VOCAL + MOVEMENT)
ev["is_target"] = ev.split.isin(["adapt", "test"])
ev = ev[ev.has_audio.astype(bool)].reset_index(drop=True)
for c in ["correct_audio", "correct_camera"]:
    ev[c] = ev[c].astype(bool).astype(float)
print(f"Clips with audio: {len(ev)} (source {(~ev.is_target).sum()}, target {ev.is_target.sum()}), "
      f"sites {ev.location_id.nunique()}")


# Within-species gain (high - low sound group), weighted by group sizes
def adjusted_gain(d, hi, sensor):
    g = pd.DataFrame({"sp": d.species.to_numpy(), "hi": hi, "ok": d[f"correct_{sensor}"].to_numpy()})
    a = g.groupby(["sp", "hi"]).ok.agg(["mean", "size"]).unstack("hi")
    a = a[(a["size"][True] >= MIN_PER_GROUP) & (a["size"][False] >= MIN_PER_GROUP)] if True in a["size"] else a.iloc[:0]
    if a.empty:
        return np.nan, 0
    w = 1 / (1 / a["size"][True] + 1 / a["size"][False])
    return float(((a["mean"][True] - a["mean"][False]) * w).sum() / w.sum()), len(a)


def contrasts(d, hi):
    raw_a = d.correct_audio[hi].mean() - d.correct_audio[~hi].mean()
    raw_c = d.correct_camera[hi].mean() - d.correct_camera[~hi].mean()
    adj_a, n_sp = adjusted_gain(d, hi, "audio")
    adj_c, _ = adjusted_gain(d, hi, "camera")
    return {"audio acc, high group": d.correct_audio[hi].mean(), "audio acc, low group": d.correct_audio[~hi].mean(),
            "audio gain, raw": raw_a, "camera gain, raw": raw_c,
            "audio gain, species-adjusted (PRIMARY)": adj_a, "camera gain, species-adjusted": adj_c,
            "audio - camera gain, species-adjusted": adj_a - adj_c}, n_sp


def run(d, sample, score_name, grouping):
    s = d[score_name].to_numpy()
    cut = np.quantile(s, 1 - TOP_SHARE) if grouping == "top 25%" else CLEAR
    hi_all = s >= cut
    point, n_sp = contrasts(d, hi_all)
    sites = d.location_id.to_numpy()
    u = np.unique(sites)
    by_site = {k: np.flatnonzero(sites == k) for k in u}
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in point}
    for _ in range(N_BOOT):
        rows = np.concatenate([by_site[k] for k in rng.choice(u, len(u))])
        b, hi = d.iloc[rows], hi_all[rows]
        if hi.sum() == 0 or (~hi).sum() == 0:
            continue
        for k, v in contrasts(b, hi)[0].items():
            boot[k].append(v)
    out = []
    for k, v in point.items():
        bv = np.asarray(boot[k], float)
        bv = bv[~np.isnan(bv)]
        lo, hi_ = (np.percentile(bv, [2.5, 97.5]) if len(bv) > 50 else (np.nan, np.nan))
        out.append({"sample": sample, "score": score_name, "grouping": grouping, "cut": round(float(cut), 3),
                    "quantity": k, "value_pts": 100 * v, "lo_pts": 100 * lo, "hi_pts": 100 * hi_,
                    "n_high": int(hi_all.sum()), "n_low": int((~hi_all).sum()),
                    "n_species_adjusted": n_sp, "n_sites": len(u)})
    return out


# Pooled, target only and source only; top 25% or score >= 0.3
samples = {"pooled": ev, "target": ev[ev.is_target], "source (out-of-fold)": ev[~ev.is_target]}
rows = []
for sname, d in samples.items():
    d = d.reset_index(drop=True)
    for score_name in ["call_score", "animal_score"]:
        for grouping in ["top 25%", "score >= 0.3"]:
            rows += run(d, sname, score_name, grouping)
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "audio_call_check.csv", index=False)

# Accuracy by fifth of the call score
ev["quintile"] = pd.qcut(ev.call_score.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
q = ev.groupby("quintile", observed=True).agg(n=("species", "size"), call_score_min=("call_score", "min"),
                                              call_score_max=("call_score", "max"),
                                              audio_acc=("correct_audio", "mean"),
                                              camera_acc=("correct_camera", "mean"),
                                              share_birds=("class", lambda x: (x == "Aves").mean()))
q.to_csv(RESULTS / "audio_call_check_quintiles.csv")

pd.set_option("display.width", 220)
show = lambda r: f"{r.value_pts:+6.1f} [{r.lo_pts:+.1f}, {r.hi_pts:+.1f}]"
print(f"\n=== Who is in the high-call group (pooled, top 25%, cut {res.iloc[0].cut}) ===")
hi = ev.call_score >= res.iloc[0].cut
print(f"birds: {hi[ev['class'] == 'Aves'].mean():.0%} of bird clips | mammals: {hi[ev['class'] == 'Mammalia'].mean():.0%}"
      f" of mammal clips | clips with score >= 0.3: {(ev.call_score >= CLEAR).sum()}")
for (sname, sc, gr), g in res.groupby(["sample", "score", "grouping"], sort=False):
    r0 = g.iloc[0]
    print(f"\n--- {sname} | {sc} | {gr} (cut {r0.cut}) | high {r0.n_high} / low {r0.n_low} clips | "
          f"{r0.n_species_adjusted} species in adjusted gain | {r0.n_sites} sites ---")
    for r in g.itertuples():
        print(f"  {r.quantity:42s} {show(r)}")
print("\n=== Dose-response: accuracy by call-score quintile (pooled) ===")
print(q.round(3).to_string())

# Pre-registered verdict on the pooled call score, top 25%
p = res[(res["sample"] == "pooled") & (res.score == "call_score") & (res.grouping == "top 25%")].set_index("quantity")
a, dd = p.loc["audio gain, species-adjusted (PRIMARY)"], p.loc["audio - camera gain, species-adjusted"]
if a.lo_pts > 0 and dd.lo_pts > 0:
    verdict = "audio works when there is a sound -> the data (few calls) is the main limit"
elif not (a.lo_pts > 0):
    verdict = "audio is weak even when a sound is present -> the model is a limit; prioritise the Perch check"
else:
    verdict = "high-score clips are easier for both sensors -> inconclusive about sound"
print(f"\nPRIMARY (pooled, call score, top 25%): audio gain {show(a)} | audio - camera {show(dd)}")
print(f"Pre-registered verdict: {verdict}")
print(f"Saved {RESULTS / 'audio_call_check.csv'} and {RESULTS / 'audio_call_check_quintiles.csv'}")

# Fig 7: accuracy by fifth of the call score
qf = pd.read_csv(RESULTS / "audio_call_check_quintiles.csv")
fig, ax = plt.subplots(figsize=(W, 2.4))
for col_name, lab, col in [("camera_acc", "Camera", CAMERA), ("audio_acc", "Audio", AUDIO)]:
    ax.plot(qf.quintile, 100 * qf[col_name], color=col, lw=2, marker="o", ms=5, label=lab)
    for xi, v in zip(qf.quintile, 100 * qf[col_name]):
        ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7,
                    color=INK2)
ax.set_xticks(qf.quintile, [f"Q{i}" for i in qf.quintile])
ax.set_xlabel("Fifth of the AudioSet call score (Q1 = least animal sound)")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 70)
ax.legend(loc="lower right")
save(fig, "fig7_call_quintiles")
