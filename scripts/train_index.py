# Streams the large train annotation JSON into train_index.csv (one row per video)
import time
from collections import defaultdict

import ijson
import pandas as pd

from config import ANN_TRAIN, OUT


# The JSON contains NaN, which ijson cannot parse, so it is replaced with null while streaming
class NaNToNull:
    def __init__(self, f):
        self.f, self.buf = f, b""
    def read(self, n=1 << 20):
        chunk = self.f.read(n)
        data = (self.buf + chunk).replace(b"NaN", b"null")
        keep = 0
        if chunk:
            keep = 2 if data.endswith(b"Na") else 1 if data.endswith(b"N") else 0
        self.buf = data[len(data) - keep:] if keep else b""
        return data[:len(data) - keep]


def stream(prefix):
    with open(ANN_TRAIN, "rb") as f:
        yield from ijson.items(NaNToNull(f), prefix)


t0 = time.time()
videos = {}
for v in stream("videos.item"):
    fps = float(v["video_fps"]) if v.get("video_fps") else None
    videos[v["id"]] = {
        "video_name": v["video_name"],
        "location_id": v.get("location_id"),
        "duration_s": float(v["video_num_frames"]) / fps if fps else None,
    }
print(f"videos: {len(videos)} ({time.time()-t0:.0f}s)")

species = defaultdict(set)
for a in stream("annotations.item"):
    species[a["video_id"]].add(a["noun_phrase"])
print(f"annotation pass done ({time.time()-t0:.0f}s)")

rows = []
for vid, v in videos.items():
    sp = sorted(species.get(vid, []))
    rows.append({**v, "species_list": "|".join(sp), "n_species": len(sp),
                 "species": sp[0] if len(sp) == 1 else ""})
pd.DataFrame(rows).to_csv(OUT / "train_index.csv", index=False)
print(f"Saved {OUT / 'train_index.csv'} ({len(rows)} rows, {time.time()-t0:.0f}s)")
