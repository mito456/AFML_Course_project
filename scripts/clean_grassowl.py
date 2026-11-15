"""Clean the Taiwan Grass-Owl paired camera/recorder detections.

Inputs : data/raw/grassowl_taiwan/Data_Hour_combine.csv, Data_Day_combine.csv
Outputs: data/processed/grassowl_hourly.csv, grassowl_daily.csv

Source: Chang, Hong & Tsai (2026), Mendeley Data, doi:10.17632/tdnpwgv6s5.1, CC BY 4.0.
These are the effort-standardized files (7 days/month, night hours 17:00-06:00;
recorder 10 min/hour per the readme). C = perch camera trap, R = acoustic recorder.

Usage: python scripts/clean_grassowl.py [--raw DIR] [--out DIR]
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Inferred from station prefixes and the dataset description (two river systems).
# ZW/YS are clearly Zengwen-Yanshui; GR/JR/LG assumed Kaoping. Confirm with authors.
RIVER = {"ZW": "Zengwen-Yanshui", "YS": "Zengwen-Yanshui",
         "GR": "Kaoping", "JR": "Kaoping", "LG": "Kaoping"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw/grassowl_taiwan")
    p.add_argument("--out", type=Path, default=ROOT / "data/processed")
    return p.parse_args()


def outcome(c, r):
    if pd.isna(c) or pd.isna(r):
        return pd.NA
    return {(1, 1): "both", (1, 0): "camera_only", (0, 1): "recorder_only", (0, 0): "neither"}[(int(c), int(r))]


def tidy(df, hourly):
    out = pd.DataFrame({
        "station": df["station"],
        "river_system": df["station"].str[:2].map(RIVER),
        "date": pd.to_datetime(df[["year", "month", "day"]]).dt.date,
    })
    if hourly:
        out["hour"] = df["hour"]
        # Night named by its evening date: hours 0-6 belong to the previous evening.
        ts = pd.to_datetime(out["date"]) + pd.to_timedelta(out["hour"], unit="h")
        out["night"] = (ts - pd.Timedelta(hours=12)).dt.date
    out["month"] = df["month"]
    out["campaign"] = pd.to_datetime(out["date"]).apply(
        lambda d: "2023-24" if d < pd.Timestamp("2024-02-15") else "2024-25")
    out["breeding"] = df["breeding_state"].map({"Y": 1, "N": 0})
    out["camera"] = pd.to_numeric(df["C-presence"], errors="coerce").astype("Int64")
    out["recorder"] = pd.to_numeric(df["R-presence"], errors="coerce").astype("Int64")
    out["outcome"] = [outcome(c, r) for c, r in zip(out["camera"], out["recorder"])]
    return out


def main():
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    h_src = pd.read_csv(a.raw / "Data_Hour_combine.csv")
    d_src = pd.read_csv(a.raw / "Data_Day_combine.csv")
    h, d = tidy(h_src, True), tidy(d_src, False)

    assert len(h) == len(h_src) and len(d) == len(d_src), "row count changed"
    assert h["river_system"].notna().all(), "unmapped station prefix"
    assert not h.duplicated(["station", "date", "hour"]).any(), "duplicate station-hour"
    assert not d.duplicated(["station", "date"]).any(), "duplicate station-day"

    # Flag daily rows that disagree with the hourly data aggregated to that day.
    hd = (h.groupby(["station", "date"])[["camera", "recorder"]]
            .max().rename(columns={"camera": "camera_from_hourly", "recorder": "recorder_from_hourly"})
            .reset_index())
    d = d.merge(hd, on=["station", "date"], how="left")
    d["agrees_with_hourly"] = ((d["camera"] == d["camera_from_hourly"]) &
                               (d["recorder"] == d["recorder_from_hourly"])).astype("Int64")
    d.loc[d["camera_from_hourly"].isna() | d["recorder_from_hourly"].isna(), "agrees_with_hourly"] = pd.NA

    h.to_csv(a.out / "grassowl_hourly.csv", index=False)
    d.to_csv(a.out / "grassowl_daily.csv", index=False)

    print(f"wrote grassowl_hourly.csv: {len(h)} rows; grassowl_daily.csv: {len(d)} rows")
    print(f"stations: {h.station.nunique()}; range {h.date.min()} to {h.date.max()}")
    print(f"hourly missing camera: {h.camera.isna().sum()}, missing recorder: {h.recorder.isna().sum()}")
    print("hourly outcomes:", h.outcome.value_counts().to_dict())
    print("daily vs hourly:", d.agrees_with_hourly.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
