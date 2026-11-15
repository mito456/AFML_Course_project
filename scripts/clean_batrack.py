"""Clean the BatRack 2020 audio-triggered clip table into one tidy paired table.

Input : data/raw/batrack2020/audio_trigger_performance_data.csv
Output: data/processed/batrack_pairs.csv

Each row is one audio-triggered recording at station BR01 with a matched video
clip and a matched audio clip, each labelled by the BatRack authors.

Conventions found in the source files (checked against Fig6_data.csv):
- File-name times (video and audio) are local time, Europe/Berlin (CEST in summer).
- The source `timestamp` column is when the clip was labelled, not recorded; dropped.
- Sunset/sunrise in the source are UTC. We recompute them with astral at
  SITE_LAT/SITE_LON, fitted to the source's sunset/sunrise (mean error ~2.5 s).

Usage: python scripts/clean_batrack.py [--raw DIR] [--out FILE] [--max-gap SECONDS]
"""
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
from astral import LocationInfo
from astral.sun import sun

SITE_LAT, SITE_LON = 50.90, 8.68  # fitted to BatRack Fig6 sunset/sunrise times
TZ = "Europe/Berlin"
ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw/batrack2020")
    p.add_argument("--out", type=Path, default=ROOT / "data/processed/batrack_pairs.csv")
    p.add_argument("--max-gap", type=float, default=5.0,
                   help="max seconds between video and audio file times for pair_ok")
    return p.parse_args()


def video_label(v):
    """video_bat: y -> 1, n -> 0, anything else (e, blank) -> missing."""
    v = str(v).strip().lower() if pd.notna(v) else ""
    return {"y": 1, "n": 0}.get(v, pd.NA)


def audio_labels(note):
    """Notes column: 'n' = no calls; 'y<code>' = calls, code = species/call type.

    Returns (audio_bat, audio_uncertain, audio_code).
    """
    if pd.isna(note) or not str(note).strip():
        return pd.NA, pd.NA, pd.NA
    s = str(note).strip().lower()
    if s == "n":
        return 0, 0, pd.NA
    if s.startswith("y"):
        code = s[1:]
        uncertain = int("?" in code)
        code = code.replace("?", "")
        return 1, uncertain, (code or pd.NA)
    return pd.NA, pd.NA, s


def sun_times(night_dates):
    obs = LocationInfo("site", "DE", "UTC", SITE_LAT, SITE_LON).observer
    rows = []
    for d in night_dates:
        rows.append({
            "night": d,
            "sunset_utc": pd.Timestamp(sun(obs, date=d)["sunset"]),
            "sunrise_utc": pd.Timestamp(sun(obs, date=d + dt.timedelta(days=1))["sunrise"]),
        })
    return pd.DataFrame(rows)


def main():
    a = parse_args()
    src = pd.read_csv(a.raw / "audio_trigger_performance_data.csv")
    n_in = len(src)

    # Recording times from file names (local time).
    v_local = pd.to_datetime(src["video_file"].str.extract(r"_(\d{8}_\d{6})\.mp4$")[0],
                             format="%Y%m%d_%H%M%S")
    a_local = pd.to_datetime(src["audio_file"].str.replace(".wav", "", regex=False),
                             format="%Y-%m-%dT%H_%M_%S")
    rec_local = a_local.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")

    audio = src["Notes"].apply(audio_labels).apply(pd.Series)
    audio.columns = ["audio_bat", "audio_uncertain", "audio_code"]

    df = pd.DataFrame({
        "clip_id": src["audio_file"].str.replace(".wav", "", regex=False),
        "station": src["station"].str.replace("_audio_audiotrigger", "", regex=False),
        "rec_time_local": rec_local,
        "rec_time_utc": rec_local.dt.tz_convert("UTC"),
        "video_path": "matched_videos/" + src["video_file"],
        "audio_path": "audio_trigger/" + src["audio_file"],
        "pair_gap_s": (v_local - a_local).dt.total_seconds().abs(),
        "video_bat": src["video_bat"].apply(video_label),
        "video_bat_raw": src["video_bat"],
        "audio_bat": audio["audio_bat"],
        "audio_uncertain": audio["audio_uncertain"],
        "audio_code": audio["audio_code"],
        "audio_note_raw": src["Notes"],
        "social_call": src["social"].map({"y": 1, "n": 0}),
        "behaviour": src["behaviour"].str.strip().replace("", pd.NA),
        "n_bats": src["number_of_bats"],
    })
    df["pair_ok"] = (df["pair_gap_s"] <= a.max_gap).astype(int)

    # Context features. A "night" is named by its evening date (local time - 12 h).
    df["night"] = (df["rec_time_local"] - pd.Timedelta(hours=12)).dt.date
    df["hour_local"] = df["rec_time_local"].dt.hour
    st = sun_times(sorted(df["night"].dropna().unique()))
    df = df.merge(st, on="night", how="left")
    df["mins_since_sunset"] = ((df["rec_time_utc"] - df["sunset_utc"]).dt.total_seconds() / 60).round(1)
    df["mins_to_sunrise"] = ((df["sunrise_utc"] - df["rec_time_utc"]).dt.total_seconds() / 60).round(1)
    df["is_dark"] = ((df["mins_since_sunset"] >= 0) & (df["mins_to_sunrise"] >= 0)).astype(int)

    # Outcome category for complementarity counts.
    both = df[["video_bat", "audio_bat"]]
    df["outcome"] = pd.NA
    df.loc[(both["video_bat"] == 1) & (both["audio_bat"] == 1), "outcome"] = "both"
    df.loc[(both["video_bat"] == 1) & (both["audio_bat"] == 0), "outcome"] = "video_only"
    df.loc[(both["video_bat"] == 0) & (both["audio_bat"] == 1), "outcome"] = "audio_only"
    df.loc[(both["video_bat"] == 0) & (both["audio_bat"] == 0), "outcome"] = "neither"

    df = df.sort_values("rec_time_utc").reset_index(drop=True)

    # Checks.
    assert len(df) == n_in, "row count changed"
    assert df["clip_id"].is_unique, "duplicate clip_id"
    assert df["rec_time_local"].notna().all(), "unparsed recording time"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    usable = df[(df.pair_ok == 1) & df.outcome.notna()]
    print(f"wrote {a.out.relative_to(ROOT)}: {len(df)} rows, {df.shape[1]} columns")
    print(f"nights: {df.night.nunique()} ({df.night.min()} to {df.night.max()})")
    print(f"pairs with gap > {a.max_gap:g}s: {(df.pair_ok == 0).sum()}")
    print(f"missing video label: {df.video_bat.isna().sum()}, missing audio label: {df.audio_bat.isna().sum()}")
    print("outcomes (usable pairs):", usable.outcome.value_counts().to_dict())
    print("audio codes:", df.audio_code.value_counts().to_dict())


if __name__ == "__main__":
    main()
