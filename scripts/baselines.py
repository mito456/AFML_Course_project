# Global camera and audio experts: trained on source sites, tested on the 525 target clips
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import camera_expert as cam
from config import OUT, RESULTS

# AudioSet labels used to describe the sound in each clip (analysis only)
MAMMAL_SOUND = ["Grunt", "Snort", "Whimper", "Walk, footsteps", "Chewing, mastication", "Chatter", "Growling",
                "Roar", "Roaring cats (lions, tigers)", "Pig", "Oink", "Squeal", "Hiss", "Purr", "Bark", "Howl",
                "Rodents, rats, mice", "Rustle", "Squeak", "Scratch", "Wild animals"]
BIRD_SOUND = ["Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", "Squawk", "Fowl", "Chicken, rooster",
              "Cluck", "Turkey", "Gobble", "Coo", "Pigeon, dove", "Crow", "Caw", "Hoot",
              "Bird flight, flapping wings", "Walk, footsteps", "Rustle", "Wild animals"]
NOISE_SOUND = ["Wind", "Wind noise (microphone)", "Rustling leaves", "Rain", "Raindrop", "Rain on surface",
               "Thunderstorm", "Thunder", "Stream", "Water", "Waterfall", "Insect", "Cricket", "Mosquito",
               "Fly, housefly", "Bee, wasp, etc.", "Frog", "Croak", "Noise", "Environmental noise", "White noise",
               "Silence"]


# Audio expert: class-balanced logistic regression on AST embeddings
def fit_proba(X_tr, y_tr, X_te, classes):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    m.fit(X_tr, y_tr)
    p = np.zeros((len(X_te), len(classes)))
    p[:, [classes.index(c) for c in m.classes_]] = m.predict_proba(X_te)
    return p


def macro_f1(y, p, classes):
    return f1_score(y, np.array(classes)[p.argmax(1)], labels=classes, average="macro", zero_division=0)


idx = pd.read_csv(OUT / "feat_index.csv")
V, A, P = np.load(OUT / "visual_emb.npy"), np.load(OUT / "audio_emb.npy"), np.load(OUT / "audioset_probs.npy")
labels = pd.read_csv(OUT / "audioset_labels.csv").label.tolist()
meta = pd.read_csv(OUT / "test_meta.csv").set_index("video_name")

classes = sorted(idx.species.unique())
src = (idx.split == "source").to_numpy()
tgt = idx.split.isin(["adapt", "test"]).to_numpy()
has_a = idx.has_audio.astype(bool).to_numpy()
y = idx.species.to_numpy()
print(f"Features: {len(idx)} videos | train (source sites) {src.sum()} | test (target sites) {tgt.sum()} | species {len(classes)}")
if len(idx) < len(pd.read_csv(OUT / "split.csv")):
    print("NOTE: partial feature set -> numbers below are only a smoke test.")

frames = cam.load_frames(idx)
p_vis = cam.fit_proba(frames, y[src], np.flatnonzero(src), np.flatnonzero(tgt), classes)
# Audio is trained on clips with audio; clips without audio get uniform probabilities
tr_a = src & has_a
p_aud = fit_proba(A[tr_a], y[tr_a], A[tgt], classes)
p_aud[~has_a[tgt]] = 1.0 / len(classes)

models = {"global_visual": p_vis, "audio_only": p_aud}
yt = y[tgt]
chance = 1 / len(classes)
summary = pd.DataFrame([{"model": k, "macro_f1": macro_f1(yt, p, classes),
                         "accuracy": accuracy_score(yt, np.array(classes)[p.argmax(1)])} for k, p in models.items()])
print(f"\n=== Test clips at unseen camera sites (chance accuracy = {chance:.2f}) ===")
print(summary.round(3).to_string(index=False))
summary.to_csv(RESULTS / "results_summary.csv", index=False)

per_sp = pd.DataFrame({k: f1_score(yt, np.array(classes)[p.argmax(1)], labels=classes, average=None, zero_division=0)
                       for k, p in models.items()}, index=classes)
per_sp["n_test"] = pd.Series(yt).value_counts()
per_sp["audio_minus_visual"] = per_sp.audio_only - per_sp.global_visual
per_sp.round(3).to_csv(RESULTS / "results_per_species.csv")
print("\n=== Per-species F1 (sorted by where audio does best relative to visual) ===")
print(per_sp.round(2).sort_values("audio_minus_visual", ascending=False).to_string())

# Per-clip predictions with analysis-only metadata
t = idx[tgt].reset_index(drop=True)
t = t.join(meta[["is_ir", "class", "visible_frac", "mean_area", "placement", "exits_frame"]], on="video_name")
for k, p in models.items():
    t[f"pred_{k}"] = np.array(classes)[p.argmax(1)]
    t[f"conf_{k}"] = p.max(1)
    t[f"correct_{k}"] = t[f"pred_{k}"] == t.species
Pt = P[tgt]
col = {l: i for i, l in enumerate(labels)}
mx = lambda names: Pt[:, [col[n] for n in names if n in col]].max(1)
mam, bird, noise = mx(MAMMAL_SOUND), mx(BIRD_SOUND), mx(NOISE_SOUND)
is_bird = (t["class"] == "Aves").to_numpy()
t["target_sound"] = np.where(is_bird, bird, mam)
t["offtarget_sound"] = np.where(is_bird, mam, bird)
t["noise_sound"] = noise
t.to_csv(OUT / "predictions.csv", index=False)
np.savez(OUT / "predictions.npz", classes=np.array(classes), video_name=t.video_name.to_numpy(),
         global_visual=p_vis, audio_only=p_aud, visual_emb=V[tgt], audio_emb=A[tgt])

# Accuracy by condition
small_cut = t.mean_area.quantile(0.25)
slices = {
    "night (IR)": t.is_ir == True, "day (colour)": t.is_ir == False,
    "bird": t["class"] == "Aves", "mammal": t["class"] == "Mammalia",
    "small animal (bottom 25% area)": t.mean_area <= small_cut, "larger animal": t.mean_area > small_cut,
    "brief visibility (<50% frames)": t.visible_frac < 0.5, "visible most of clip": t.visible_frac >= 0.5,
    "arboreal camera": t.placement == "arboreal", "terrestrial camera": t.placement == "terrestrial",
    "has audio": t.has_audio == 1, "no audio": t.has_audio == 0,
    "target sound likely (top 25%)": t.target_sound >= t.target_sound.quantile(0.75),
    "target sound unlikely": t.target_sound < t.target_sound.quantile(0.75),
}
rows = [{"slice": n, "n": int(m.sum()), **{k: t.loc[m, f"correct_{k}"].mean() for k in models}}
        for n, m in slices.items() if m.sum() > 0]
sl = pd.DataFrame(rows)
sl.to_csv(RESULTS / "results_slices.csv", index=False)
print("\n=== Accuracy by condition ===")
print(sl.round(2).to_string(index=False))

# Which sensor gets each clip right
both = (t.correct_global_visual & t.correct_audio_only).sum()
vis_only = (t.correct_global_visual & ~t.correct_audio_only).sum()
aud_only = (~t.correct_global_visual & t.correct_audio_only).sum()
neither = (~t.correct_global_visual & ~t.correct_audio_only).sum()
print(f"\nWho gets each clip right (of {len(t)}): both {both} | visual only {vis_only} | "
      f"audio only {aud_only} | neither {neither}")
print(f"Saved results_*.csv, predictions.csv, predictions.npz to {OUT}")
