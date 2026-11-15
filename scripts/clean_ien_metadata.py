"""Flatten the India Ecoacoustics Network (IEN 2026) metadata into one file table.

Inputs : data/raw/ien_india2026/type_A_metadata.csv, type_B_metadata.csv
Output : data/processed/ien_files.csv  (one row per audio file, or file x species for Type B)

Source: Ramesh et al. (2026), A large-scale crowd-sourced annotated acoustic dataset
of Indian fauna, bioRxiv doi:10.64898/2026.07.20.739496. Data on Zenodo
(Type A 18743214, Type B 18927866), CC BY-NC: non-commercial use only.

Audio is not in the repo. Use `archive_name` to pick which state zip to download.
The metadata has a recording date but no time of day.

Usage: python scripts/clean_ien_metadata.py
"""
import ast
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ien_india2026"
OUT = ROOT / "data/processed/ien_files.csv"


def lit(s):
    try:
        return ast.literal_eval(s) if isinstance(s, str) else {}
    except (ValueError, SyntaxError):
        return {}


def notes(s):
    try:
        n = json.loads(s) if isinstance(s, str) else {}
    except json.JSONDecodeError:
        return {}
    return n if isinstance(n, dict) else {}


def tidy(path, label_type):
    d = pd.read_csv(path)
    taxa, media, nts = d.taxa_info.apply(lit), d.media_info.apply(lit), d.notes.apply(notes)
    out = pd.DataFrame({
        "media_file_name": d.media_file_name,
        "label_type": label_type,  # A = strong (boxes in annotations.csv), B = weak (file-level)
        "archive_name": d.archive_name,
        "state": d.archive_name.str.extract(r"_type_[AB]_(.+)\.zip$")[0],
        "recording_date": pd.to_datetime(d.recording_date, errors="coerce").dt.date,
        "latitude": d.latitude,
        "longitude": d.longitude,
        "species": taxa.apply(lambda t: t.get("canonicalName")),
        "taxon_rank": taxa.apply(lambda t: t.get("rank")),
        "class": taxa.apply(lambda t: t.get("class")),
        "order": taxa.apply(lambda t: t.get("order")),
        "duration_s": media.apply(lambda m: m.get("media_length_seconds")),
        "sample_rate_khz": media.apply(lambda m: m.get("sampling_rate_KHz")),
        "size_mb": media.apply(lambda m: m.get("byte_size_Mb")),
        "habitat": nts.apply(lambda n: (n.get("habitat") or "").strip() or pd.NA),
        "authors": d.author_names.str.strip(),
    })
    if "validation" in d:
        out["validation"] = d.validation
    return out


def main():
    a = tidy(RAW / "type_A_metadata.csv", "A")
    b = tidy(RAW / "type_B_metadata.csv", "B")
    df = pd.concat([a, b], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    files = df.drop_duplicates(["label_type", "media_file_name"])
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} rows")
    for t, g in files.groupby("label_type"):
        print(f"type {t}: {len(g)} files, {g.duration_s.sum()/60:.0f} min, {g.state.nunique()} states, "
              f"dates {g.recording_date.min()} to {g.recording_date.max()}")
    print("classes:", df["class"].value_counts().to_dict())


if __name__ == "__main__":
    main()
