# Detects infrared/greyscale footage from frame colourfulness
import shutil, subprocess, numpy as np


# Uses system ffmpeg, else the one bundled with imageio-ffmpeg
def _ffmpeg():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _ffmpeg()


# Median colour difference over 3 frames; low values mean IR/greyscale
def colorfulness(path, dur, w=64, h=36):
    vals = []
    for t in (0.25, 0.5, 0.75):
        raw = subprocess.run(
            [FFMPEG, "-v", "error", "-ss", f"{dur*t:.2f}", "-i", str(path), "-frames:v", "1",
             "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True).stdout
        if len(raw) == w * h * 3:
            a = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(float)
            vals.append(np.mean(np.abs(a[:, 0] - a[:, 1]) + np.abs(a[:, 1] - a[:, 2])))
    return float(np.median(vals)) if vals else np.nan


# Computes colourfulness for videos not yet in the cache and appends them
def fill_cache(names, durations, cache_path, budget_s=None, video_dir=None):
    import time, pandas as pd
    from pathlib import Path
    from config import VIDEO_DIR
    VIDEO_DIR = Path(video_dir) if video_dir else VIDEO_DIR
    cache_path = Path(cache_path)
    done = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["video_name", "colorfulness"])
    have = set(done.video_name)
    todo = [(n, d) for n, d in zip(names, durations) if n not in have]
    t0, new = time.time(), []
    for n, d in todo:
        if budget_s and time.time() - t0 > budget_s:
            break
        new.append({"video_name": n, "colorfulness": colorfulness(VIDEO_DIR / f"{n}.mp4", d)})
    if new:
        pd.concat([done, pd.DataFrame(new)]).to_csv(cache_path, index=False)
    return len(todo) - len(new)
