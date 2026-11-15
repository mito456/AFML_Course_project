# Picks the species and builds the source/target split from the official location-disjoint split
import numpy as np
import pandas as pd

from config import OUT, ROOT, SEED

# 23 species, at most 80 source clips each, 3 adapt clips per species in the target
N_SPECIES, MAX_TRAIN, K_ADAPT = 23, 80, 3
# Catch-all labels
EXCLUDE = {"bird", "lizard", "a monkey"}
# Min number of videos and sites, so the model doesn't learn the background instead of the animal
MIN_TRAIN_V, MIN_TRAIN_S, MIN_TEST_V, MIN_TEST_S = 40, 5, 10, 3


def clean(d):
    return d[(d.n_species == 1) & d.location_id.notna() & ~d.species.isin(EXCLUDE)]


# Samples spread over sites (one per site in turn) instead of many from one site
def spread_sample(grp, n, seed):
    grp = grp.sample(frac=1, random_state=seed)
    grp = grp.assign(_rank=grp.groupby("location_id").cumcount())
    return grp.sort_values(["_rank", "location_id"]).head(n).drop(columns="_rank")


tr = clean(pd.read_csv(OUT / "train_index.csv"))
te = clean(pd.read_csv(OUT / "test_meta.csv"))

stats = (tr.groupby("species").agg(train_v=("video_name", "size"), train_s=("location_id", "nunique"))
         .join(te.groupby("species").agg(test_v=("video_name", "size"), test_s=("location_id", "nunique")), how="inner"))
ok = stats[(stats.train_v >= MIN_TRAIN_V) & (stats.train_s >= MIN_TRAIN_S)
           & (stats.test_v >= MIN_TEST_V) & (stats.test_s >= MIN_TEST_S)]
species = ok.sort_values("test_v", ascending=False).head(N_SPECIES).index.tolist()

src = pd.concat([spread_sample(g, MAX_TRAIN, SEED) for _, g in tr[tr.species.isin(species)].groupby("species")])
src = src.assign(split="source")

tgt = te[te.species.isin(species)].copy()
adapt_idx = [i for _, g in tgt.groupby("species") for i in spread_sample(g, K_ADAPT, SEED).index]
tgt["split"] = "test"
tgt.loc[adapt_idx, "split"] = "adapt"

split = pd.concat([src, tgt])[["video_name", "species", "location_id", "split"]]
split.to_csv(OUT / "split.csv", index=False)

# Maps the chosen source clips to bucket paths for download (train_list.txt)
listing = [l.strip() for l in open(ROOT / "train_files.txt") if l.strip() and not l.strip().endswith("/")]
by_name = {l.rsplit("/", 1)[-1].removesuffix(".mp4"): l for l in listing}
paths = [by_name.get(n) for n in src.video_name]
missing = [n for n, p in zip(src.video_name, paths) if p is None]
with open(OUT / "train_list.txt", "w") as f:
    f.write("\n".join(p for p in paths if p) + "\n")

tab = split.pivot_table(index="species", columns="split", values="video_name", aggfunc="size", fill_value=0)[["source", "adapt", "test"]]
tab["src_sites"] = src.groupby("species").location_id.nunique()
tab["tgt_sites"] = tgt.groupby("species").location_id.nunique()
print(tab.sort_values("test", ascending=False).to_string())
print(f"\nVideos: {tab.source.sum()} source | {tab.adapt.sum()} adapt | {tab.test.sum()} test")
print("Site overlap source/target:", len(set(src.location_id) & set(tgt.location_id)))
mean_dur = src.duration_s.mean()
print(f"Train download: {len(paths) - len(missing)} files (~{(len(paths)-len(missing)) * 10.5 / 1024:.1f} GB est.) | "
      f"not found in bucket listing: {len(missing)} {missing[:5]}")
print(f"Saved {OUT / 'split.csv'} and {OUT / 'train_list.txt'}")
