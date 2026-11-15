# Builds test_meta.csv: per-video metadata for the official test set
import json
import re
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

from config import ANN_TEST, OUT, VIDEO_DIR
from night_check import colorfulness, fill_cache


# Camera placement from the location id (arboreal / terrestrial / unknown)
def placement(loc):
    l = str(loc).lower()
    if "arboreal" in l or l.startswith("arb_"):
        return "arboreal"
    if "terrestrial" in l:
        return "terrestrial"
    return "unknown"


# Project = letters before the number in location_id; COPLA is treated as a typo of COLPA
def project(loc):
    l = str(loc)
    if l == "nan":
        return "unknown"
    if re.fullmatch(r"[0-9a-f]{8}", l):
        return "hash_site"
    m = re.match(r"[A-Za-z]+", l)
    p = m.group(0).upper() if m else "unknown"
    return "COLPA" if p == "COPLA" else p


def touches_border(b, w, h, tol=1.0):
    x, y, bw, bh = b
    return x <= tol or y <= tol or x + bw >= w - tol or y + bh >= h - tol


def audio_probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    return (1, int(out.split()[0])) if out else (0, 0)


# Load test annotations without masks to save memory
with open(ANN_TEST, "r", encoding="utf-8") as f:
    data = json.load(f)
for a in data["annotations"]:
    a.pop("segmentations", None)

cats = {c["id"]: c for c in data["categories"]}
tracks = defaultdict(list)
for a in data["annotations"]:
    tracks[a["video_id"]].append(a)

rows, bad_len = [], 0
# Visibility, box size and frame exits per video (used for analysis only, never as model inputs)
for v in data["videos"]:
    L = v["length"]
    tr = tracks.get(v["id"], [])
    species = sorted({t["noun_phrase"] for t in tr})

    visible = np.zeros(L, dtype=bool)
    areas, border, exits = [], [], False
    for t in tr:
        if len(t["bboxes"]) != L:
            bad_len += 1
        W, H = t["width"], t["height"]
        idx = [i for i, b in enumerate(t["bboxes"]) if b is not None]
        for i in idx:
            visible[i] = True
            areas.append(t["areas"][i] / (W * H))
            border.append(touches_border(t["bboxes"][i], W, H))
        if idx and idx[-1] < L - 1 and touches_border(t["bboxes"][idx[-1]], W, H):
            exits = True
    vis_idx = np.flatnonzero(visible)

    tax = {}
    if len(species) == 1:
        c = cats.get(tr[0]["category_id"], {})
        tax = {k.lower(): c.get(k) for k in ("Class", "Order", "Family")}

    ts = pd.to_datetime(v.get("video_creation_datetime"), errors="coerce")
    rows.append({
        "video_id": v["id"],
        "video_name": v["video_name"],
        "location_id": str(v.get("location_id")),
        "project": project(v.get("location_id")),
        "placement": placement(v.get("location_id")),
        "width": v["width"], "height": v["height"],
        "length_6fps": L,
        "duration_s": v["video_num_frames"] / v["video_fps"] if v.get("video_fps") else np.nan,
        "species_list": "|".join(species),
        "n_species": len(species),
        "species": species[0] if len(species) == 1 else "",
        "n_masklets": len(tr),
        "class": tax.get("class"), "order": tax.get("order"), "family": tax.get("family"),
        "visible_frac": visible.mean() if L else np.nan,
        "mean_area": float(np.mean(areas)) if areas else np.nan,
        "touches_border_frac": float(np.mean(border)) if border else np.nan,
        "exits_frame": exits,
        "first_visible_s": vis_idx[0] / 6 if len(vis_idx) else np.nan,
        "last_visible_s": vis_idx[-1] / 6 if len(vis_idx) else np.nan,
        "has_timestamp": bool(pd.notna(ts)),
    })

meta = pd.DataFrame(rows)

# Audio track presence and sample rate
print("Probing audio tracks (1-2 min)...")
probe = [audio_probe(VIDEO_DIR / f"{n}.mp4") for n in meta["video_name"]]
meta["has_audio"] = [p[0] for p in probe]
meta["audio_sr"] = [p[1] for p in probe]

# Infrared footage = low colourfulness
print("Checking frames for infrared/grayscale footage (~5 min)...")
cache = OUT / "ir_cache.csv"
fill_cache(meta.video_name, meta.duration_s, cache)
meta = meta.merge(pd.read_csv(cache), on="video_name", how="left")
meta["is_ir"] = meta["colorfulness"] < 4.0

meta.to_csv(OUT / "test_meta.csv", index=False)

single = meta[meta.n_species == 1]
print(f"\nRows: {len(meta)} | distinct species: {meta.species_list.str.split('|').explode().nunique()}")
print(f"Single-species videos: {len(single)} | multi-species: {(meta.n_species > 1).sum()} | no tracks: {(meta.n_species == 0).sum()}")
print(f"Tracks with wrong list length: {bad_len}")
print(f"Missing taxonomy (single-species): class {single['class'].isna().sum()}, family {single['family'].isna().sum()}")
print(f"\nhas_audio=0: {(meta.has_audio == 0).sum()} | sample rates: {meta.audio_sr.value_counts().to_dict()}")
print(f"Placement: {meta.placement.value_counts().to_dict()}")
print(f"Project: {meta.project.value_counts().to_dict()}")
print(f"Has timestamp: {meta.has_timestamp.sum()} (too few for time/season analysis)")
print(f"Infrared (night) footage: {meta.is_ir.sum()} | colour: {(~meta.is_ir).sum()} | unreadable: {meta.colorfulness.isna().sum()}")
print(f"Ambiguous colourfulness (1.6-12): {((meta.colorfulness > 1.6) & (meta.colorfulness < 12)).sum()}")
print(f"Birds (single-species): {(single['class'] == 'Aves').sum()}")
print(f"exits_frame: {meta.exits_frame.sum()} | visible_frac < 0.3: {(meta.visible_frac < 0.3).sum()}")

v312 = meta[meta.video_id == 312].iloc[0]
print(f"\nCheck video 312: {v312.species}, visible_frac={v312.visible_frac:.2f} "
      f"({int(round(v312.visible_frac * v312.length_6fps))}/{v312.length_6fps}), exits_frame={v312.exits_frame}")
print(f"\nSaved {OUT / 'test_meta.csv'}")
