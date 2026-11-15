# Extracts BioCLIP embeddings of every frame at 1 fps (input to the camera expert)
import argparse
import time

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from config import OUT, ROOT

FPS_KEEP = 1.0
BATCH = 64
CACHE = OUT / "features_1fps"
CACHE.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def video_path(name, split):
    folder = "train_videos" if split == "source" else "test_videos"
    return ROOT / folder / f"{name}.mp4"


# Keeps one frame per second
def read_frames_1fps(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / FPS_KEEP)) if fps and fps > 0 else 18
    frames, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos (smoke test)")
    args = ap.parse_args()

    idx = pd.read_csv(OUT / "feat_index.csv")[["video_name", "split"]]
    work = idx if args.limit is None else idx.groupby("split").head(args.limit // 3 + 1).head(args.limit)
    print(f"Device: {device} | videos to process: {len(work)} of {len(idx)}")

    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    model = model.to(device).eval()

    t0, done = time.time(), 0
    for i, row in enumerate(work.itertuples()):
        cache_file = CACHE / f"{row.video_name}.npy"
        if cache_file.exists():
            continue
        frames = read_frames_1fps(video_path(row.video_name, row.split))
        embs = []
        with torch.no_grad():
            for b in range(0, len(frames), BATCH):
                x = torch.stack([preprocess(f) for f in frames[b:b + BATCH]]).to(device)
                e = model.encode_image(x).float()
                embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        np.save(cache_file, np.concatenate(embs) if embs else np.zeros((0, 512), np.float32))
        done += 1
        if done % 50 == 0:
            rate = (time.time() - t0) / done
            left = sum(not (CACHE / f"{n}.npy").exists() for n in work.video_name)
            print(f"{i + 1}/{len(work)} | {rate:.2f}s/video | ~{left * rate / 60:.0f} min left", flush=True)

    if args.limit is not None:
        n = [np.load(CACHE / f"{v}.npy").shape[0] for v in work.video_name]
        print(f"Smoke test done in {(time.time() - t0) / 60:.1f} min | frames per video: "
              f"min {min(n)}, median {int(np.median(n))}, max {max(n)} | not combined (partial run)")
        return
    combine(idx, t0)


# Concatenates all frames; offset[i]:offset[i+1] are the frames of clip i
def combine(idx, t0):
    embs, offset, empty = [], [0], []
    for v in idx.video_name:
        e = np.load(CACHE / f"{v}.npy")
        if len(e) == 0:
            empty.append(v)
        embs.append(e)
        offset.append(offset[-1] + len(e))
    np.savez(OUT / "visual_frames_1fps.npz", emb=np.concatenate(embs).astype(np.float32),
             offset=np.array(offset), video_name=idx.video_name.to_numpy())
    n = np.diff(offset)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. {len(idx)} videos, {offset[-1]} frames "
          f"(per video: min {n.min()}, median {int(np.median(n))}, max {n.max()}). "
          f"Videos with 0 frames: {len(empty)} {empty[:5]}")
    print(f"Saved {OUT / 'visual_frames_1fps.npz'}")


if __name__ == "__main__":
    main()
