# Light (IR/colour) and placement for the source clips -> source_meta.csv
import argparse
import time

import pandas as pd

from config import OUT, ROOT
from night_check import fill_cache

# Same IR threshold as the test clips
IR_THRESHOLD = 4.0
CACHE = OUT / "ir_cache_train.csv"


def placement(loc):
    l = str(loc).lower()
    if "arboreal" in l or l.startswith("arb_"):
        return "arboreal"
    if "terrestrial" in l:
        return "terrestrial"
    return "unknown"


ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

idx = pd.read_csv(OUT / "feat_index.csv")
src = idx[idx.split == "source"][["video_name", "species", "location_id", "has_audio"]]
dur = pd.read_csv(OUT / "train_index.csv")[["video_name", "duration_s"]]
src = src.merge(dur, on="video_name", how="left")
print(f"source clips {len(src)} | missing duration: {src.duration_s.isna().sum()}")
work = src if args.limit is None else src.head(args.limit)

t0 = time.time()
left = fill_cache(work.video_name, work.duration_s, CACHE, video_dir=ROOT / "train_videos")
print(f"colourfulness done for {len(work) - left} of {len(work)} in {time.time() - t0:.0f}s")

c = pd.read_csv(CACHE)
out = src.merge(c, on="video_name", how="left")
out["is_ir"] = (out.colorfulness < IR_THRESHOLD).astype("boolean")
out.loc[out.colorfulness.isna(), "is_ir"] = pd.NA
out["placement"] = out.location_id.map(placement)
done = out[out.colorfulness.notna()]
print(f"IR night {int(done.is_ir.sum())} | colour {int((~done.is_ir.astype(bool)).sum())} | "
      f"not yet computed or unreadable {out.colorfulness.isna().sum()} | "
      f"ambiguous colourfulness (1.6-12): {((done.colorfulness > 1.6) & (done.colorfulness < 12)).sum()}")
print("placement:", out.placement.value_counts().to_dict())
if args.limit is None:
    out[["video_name", "species", "location_id", "has_audio", "duration_s", "colorfulness", "is_ir", "placement"]] \
        .to_csv(OUT / "source_meta.csv", index=False)
    print("night share by species (source):")
    print(done.groupby("species").is_ir.mean().round(2).sort_values().to_string())
    print(f"Saved {OUT / 'source_meta.csv'}")
else:
    print(done[["video_name", "species", "colorfulness", "is_ir"]].to_string(index=False))
