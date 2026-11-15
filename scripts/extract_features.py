# Extracts per-clip BioCLIP (4-frame mean) and AST audio embeddings plus AudioSet probabilities
import argparse
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from config import OUT, ROOT

N_FRAMES = 4
SR = 16000
WIN_S = 10.0
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CACHE = OUT / "features"
CACHE.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Source clips are in train_videos, target clips in test_videos
def video_path(name, split):
    folder = "train_videos" if split == "source" else "test_videos"
    return ROOT / folder / f"{name}.mp4"


def read_frames(path, n=N_FRAMES):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in np.linspace(0.1, 0.9, n) * max(total - 1, 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# Mono 16 kHz waveform via ffmpeg
def read_audio(path):
    out = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(out, np.float32)


# Interleaves species so a partial run still covers every species
def balanced_order(split):
    split = split.sample(frac=1, random_state=0)
    split["_r"] = split.groupby(["species", "split"]).cumcount()
    return split.sort_values(["_r", "species", "split"]).drop(columns="_r").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos (smoke test)")
    args = ap.parse_args()

    split = balanced_order(pd.read_csv(OUT / "split.csv"))
    work = split if args.limit is None else split.head(args.limit)
    print(f"Device: {device} | videos to process: {len(work)} of {len(split)}")

    clip_model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    clip_model = clip_model.to(device).eval()
    fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device).eval()

    labels = [ast.config.id2label[i] for i in range(ast.config.num_labels)]
    pd.Series(labels, name="label").to_csv(OUT / "audioset_labels.csv", index_label="idx")

    t0, done = time.time(), 0
    for i, row in enumerate(work.itertuples()):
        cache_file = CACHE / f"{row.video_name}.npz"
        if cache_file.exists():
            continue
        path = video_path(row.video_name, row.split)

        # Camera: mean of 4 normalised BioCLIP frame embeddings
        frames = read_frames(path)
        if frames:
            with torch.no_grad():
                x = torch.stack([preprocess(f) for f in frames]).to(device)
                e = clip_model.encode_image(x).float()
                e = e / e.norm(dim=-1, keepdim=True)
                vis = e.mean(0).cpu().numpy()
        else:
            vis = np.full(512, np.nan, np.float32)

        # Audio: AST embedding averaged over 10 s windows; AudioSet probabilities = max over windows
        wav = read_audio(path)
        has_audio = int(wav.size > SR * 0.5)
        if has_audio:
            win = int(WIN_S * SR)
            chunks = [wav[s:s + win] for s in range(0, max(len(wav) - win, 0) + 1, win)] or [wav]
            embs, probs = [], []
            with torch.no_grad():
                for c in chunks:
                    inp = fe(c, sampling_rate=SR, return_tensors="pt").to(device)
                    out = ast(**inp, output_hidden_states=True)
                    embs.append(out.hidden_states[-1][:, :2].mean(1).squeeze(0).cpu().numpy())
                    probs.append(torch.sigmoid(out.logits).squeeze(0).cpu().numpy())
            aud, prob = np.mean(embs, 0), np.max(probs, 0)
            rms = float(np.sqrt(np.mean(wav ** 2)))
        else:
            aud, prob, rms = np.zeros(768, np.float32), np.zeros(len(labels), np.float32), 0.0

        np.savez(cache_file, vis=vis, aud=aud, prob=prob, has_audio=has_audio, rms=rms,
                 dur=len(wav) / SR, n_frames=len(frames))
        done += 1
        if done % 25 == 0:
            rate = (time.time() - t0) / done
            left = sum(not (CACHE / f"{n}.npz").exists() for n in work.video_name)
            print(f"{i + 1}/{len(work)} | {rate:.2f}s/video | ~{left * rate / 60:.0f} min left", flush=True)

    combine(split, t0)


# Stacks cached per-clip features into the shared arrays (row order = feat_index.csv)
def combine(split, t0=None):
    vis, aud, prob, rows = [], [], [], []
    for row in split.itertuples():
        f = CACHE / f"{row.video_name}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        vis.append(z["vis"]); aud.append(z["aud"]); prob.append(z["prob"])
        rows.append({"video_name": row.video_name, "split": row.split, "species": row.species,
                     "location_id": row.location_id, "has_audio": int(z["has_audio"]),
                     "audio_rms": float(z["rms"]), "audio_dur_s": float(z["dur"]), "n_frames": int(z["n_frames"])})
    np.save(OUT / "visual_emb.npy", np.stack(vis))
    np.save(OUT / "audio_emb.npy", np.stack(aud))
    np.save(OUT / "audioset_probs.npy", np.stack(prob))
    idx = pd.DataFrame(rows)
    idx.to_csv(OUT / "feat_index.csv", index=False)
    took = f" in {(time.time() - t0) / 60:.1f} min" if t0 else ""
    print(f"\nDone{took}. Features available for {len(idx)} of {len(split)} videos"
          f"{' (PARTIAL)' if len(idx) < len(split) else ''}.")
    print(f"Frames read: min {idx.n_frames.min()} | videos with audio: {idx.has_audio.sum()}/{len(idx)}")
    print(idx.groupby("split").has_audio.mean().round(2).to_string())


if __name__ == "__main__":
    main()
