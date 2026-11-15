# Times the camera and audio paths per clip (decode + model) and counts FLOPs
import argparse
import time

import numpy as np
import open_clip
import pandas as pd
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from config import OUT, RESULTS, SEED
from plot_utils import AUDIO, CAMERA, INK2, W, plt, save
from extract_features import SR, WIN_S, read_audio
from extract_frames_1fps import BATCH, read_frames_1fps, video_path

N_WARMUP = 5
N_BOOT = 1000
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

ap = argparse.ArgumentParser()
ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--per-species", type=int, default=5)
args = ap.parse_args()
dev = args.device
if dev == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA not available; run with --device cpu")
torch.set_grad_enabled(False)


# Wall clock; waits for the GPU to finish first
def clock():
    if dev == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


clip_model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
clip_model = clip_model.to(dev).eval()
fe = ASTFeatureExtractor.from_pretrained(AST_ID)
ast = ASTForAudioClassification.from_pretrained(AST_ID).to(dev).eval()


# Camera path: decode 1-fps frames + BioCLIP
def camera_path(path):
    t0 = clock()
    frames = read_frames_1fps(path)
    t1 = clock()
    for b in range(0, len(frames), BATCH):
        x = torch.stack([preprocess(f) for f in frames[b:b + BATCH]]).to(dev)
        e = clip_model.encode_image(x).float()
        _ = (e / e.norm(dim=-1, keepdim=True)).cpu()
    t2 = clock()
    return len(frames), t1 - t0, t2 - t1


# Audio path: decode audio + AST over 10 s windows
def audio_path(path):
    t0 = clock()
    wav = read_audio(path)
    t1 = clock()
    n_win = 0
    if wav.size > SR * 0.5:
        win = int(WIN_S * SR)
        chunks = [wav[s:s + win] for s in range(0, max(len(wav) - win, 0) + 1, win)] or [wav]
        for c in chunks:
            inp = fe(c, sampling_rate=SR, return_tensors="pt").to(dev)
            out = ast(**inp, output_hidden_states=True)
            _ = out.hidden_states[-1][:, :2].mean(1).cpu(), torch.sigmoid(out.logits).cpu()
        n_win = len(chunks)
    t2 = clock()
    return n_win, len(wav) / SR, t1 - t0, t2 - t1


# FLOPs per BioCLIP frame and per AST window
gflop_frame = gflop_window = np.nan
try:
    from torch.utils.flop_counter import FlopCounterMode
    from PIL import Image
    with FlopCounterMode(display=False) as fc:
        clip_model.encode_image(preprocess(Image.new("RGB", (1920, 1080))).unsqueeze(0).to(dev))
    gflop_frame = fc.get_total_flops() / 1e9
    with FlopCounterMode(display=False) as fc:
        ast(**fe(np.zeros(int(WIN_S * SR), np.float32), sampling_rate=SR, return_tensors="pt").to(dev),
            output_hidden_states=True)
    gflop_window = fc.get_total_flops() / 1e9
    print(f"FLOPs: BioCLIP {gflop_frame:.1f} GFLOP per frame | AST {gflop_window:.1f} GFLOP per 10 s window")
except Exception as ex:
    print(f"FLOP count unavailable ({type(ex).__name__}: {ex}); timing only")

idx = pd.read_csv(OUT / "feat_index.csv")
tgt = idx[idx.split.isin(["adapt", "test"])]
# 5 clips per species, plus 5 warm-up clips that are not timed
sample = tgt.sample(frac=1, random_state=SEED).groupby("species").head(args.per_species)
warm = tgt.drop(sample.index).sample(N_WARMUP, random_state=SEED)
print(f"device {dev} | warm-up {N_WARMUP} clips | timed {len(sample)} clips ({args.per_species} per species)")

for r in warm.itertuples():
    p = video_path(r.video_name, r.split)
    camera_path(p), audio_path(p)

rows = []
for i, r in enumerate(sample.itertuples()):
    p = video_path(r.video_name, r.split)
    n_fr, t_vdec, t_clip = camera_path(p)
    n_win, dur, t_adec, t_ast = audio_path(p)
    rows.append({"video_name": r.video_name, "species": r.species, "duration_s": dur, "n_frames": n_fr,
                 "n_windows": n_win, "video_decode_s": t_vdec, "bioclip_s": t_clip,
                 "audio_decode_s": t_adec, "ast_s": t_ast,
                 "camera_gflop": n_fr * gflop_frame, "audio_gflop": n_win * gflop_window})
    if (i + 1) % 20 == 0:
        print(f"{i + 1}/{len(sample)} clips timed", flush=True)

d = pd.DataFrame(rows)
d["camera_path_s"] = d.video_decode_s + d.bioclip_s
d["audio_path_s"] = d.audio_decode_s + d.ast_s
d.to_csv(RESULTS / f"compute_timing_{dev}.csv", index=False)

# Bootstrap over clips for the means and the audio / camera ratios
rng = np.random.default_rng(SEED)
B = rng.integers(0, len(d), (N_BOOT, len(d)))
dur = d.duration_s.clip(lower=0.1).to_numpy()
summ = []
for col in ["video_decode_s", "bioclip_s", "camera_path_s", "audio_decode_s", "ast_s", "audio_path_s",
            "camera_gflop", "audio_gflop"]:
    v = d[col].to_numpy(float)
    if np.isnan(v).all():
        continue
    means = v[B].mean(1)
    summ.append({"device": dev, "quantity": col, "n_clips": len(d), "mean": v.mean(),
                 "mean_lo": np.percentile(means, 2.5), "mean_hi": np.percentile(means, 97.5),
                 "median": np.median(v), "q25": np.percentile(v, 25), "q75": np.percentile(v, 75),
                 "per_video_second_mean": (v / dur).mean()})
for num, den, name in [("audio_path_s", "camera_path_s", "audio / camera time"),
                       ("audio_gflop", "camera_gflop", "audio / camera GFLOP")]:
    a, c = d[num].to_numpy(float), d[den].to_numpy(float)
    if np.isnan(a).all() or np.isnan(c).all():
        continue
    ratios = a[B].sum(1) / c[B].sum(1)
    summ.append({"device": dev, "quantity": name, "n_clips": len(d), "mean": a.sum() / c.sum(),
                 "mean_lo": np.percentile(ratios, 2.5), "mean_hi": np.percentile(ratios, 97.5),
                 "median": np.median(a / np.where(c > 0, c, np.nan))})
s = pd.DataFrame(summ)
s.to_csv(RESULTS / f"compute_summary_{dev}.csv", index=False)
pd.set_option("display.width", 200)
print(f"\n=== Cost per clip on {dev} (seconds or GFLOP; mean with 95% CI over clips) ===")
print(s.round(4).to_string(index=False))
print(f"clip duration: median {np.median(dur):.1f} s | frames median {d.n_frames.median():.0f} | "
      f"AST windows median {d.n_windows.median():.0f} | clips without audio {int((d.n_windows == 0).sum())}")
print(f"Saved {RESULTS / f'compute_timing_{dev}.csv'} and {RESULTS / f'compute_summary_{dev}.csv'}")

# Fig 4: compute cost per clip (uses whichever devices have been timed)
cost = {}
for dv in ["cpu", "cuda"]:
    f = RESULTS / f"compute_summary_{dv}.csv"
    if f.exists():
        cost[dv] = pd.read_csv(f).set_index("quantity")["mean"]
fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
rows = [(dv, path) for dv in ["cpu", "cuda"] if dv in cost for path in ["Camera", "Audio"]]
yy = np.arange(len(rows))[::-1]
for yi, (dv, path) in zip(yy, rows):
    c = cost[dv]
    parts = [("video_decode_s", "decode"), ("bioclip_s", "model")] if path == "Camera" else \
            [("audio_decode_s", "decode"), ("ast_s", "model")]
    col = CAMERA if path == "Camera" else AUDIO
    left = 0
    for (q, kind) in parts:
        ax.barh(yi, c[q], left=left, color=col, alpha=0.45 if kind == "decode" else 1.0, edgecolor="white",
                linewidth=1, height=0.7)
        left += c[q]
    ax.annotate(f"{left:.2f} s", (left, yi), xytext=(3, 0), textcoords="offset points", va="center",
                fontsize=7, color=INK2)
ax.set_yticks(yy, [f"{p} ({'GPU' if dv == 'cuda' else 'CPU'})" for dv, p in rows])
ax.set_xlabel("Seconds per clip (light = decoding, dark = model)")
ax.grid(axis="y", visible=False)
ax = axes[1]
c = cost.get("cuda", cost.get("cpu"))
vals = [c["camera_gflop"], c["audio_gflop"]]
ax.bar([0, 1], vals, color=[CAMERA, AUDIO], width=0.6, edgecolor="white", linewidth=1)
for xi, v in enumerate(vals):
    ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7,
                color=INK2)
ax.set_xticks([0, 1], ["Camera", "Audio"])
ax.set_ylabel("GFLOP per clip")
fig.subplots_adjust(wspace=0.45)
save(fig, "fig4_cost")
