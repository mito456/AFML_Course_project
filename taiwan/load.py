"""Load the cleaned Grass-Owl hourly table as one tidy row per station-hour.

Columns: station, river_system, campaign, datetime, night, hour, month, season,
breeding, camera_detected, recorder_detected, owl (the TARGET definition).
Rows where either sensor is missing are dropped, so the two sensors are always compared
on the same hours.

Run directly to print a summary: python -m taiwan.load
"""
import subprocess
import sys

import pandas as pd

from taiwan import config as C


def load(target: str = C.TARGET) -> pd.DataFrame:
    if not C.HOURLY_CSV.exists():  # rebuild from raw if the processed file is missing
        subprocess.run([sys.executable, str(C.ROOT / "scripts/clean_grassowl.py")], check=True)
    h = pd.read_csv(C.HOURLY_CSV)
    n_all = len(h)
    h = h.dropna(subset=["camera", "recorder"]).copy()
    h["camera_detected"] = h["camera"].astype(int)
    h["recorder_detected"] = h["recorder"].astype(int)
    h["datetime"] = pd.to_datetime(h["date"]) + pd.to_timedelta(h["hour"], unit="h")
    h["season"] = h["month"].map(lambda m: "wet" if m in C.WET_MONTHS else "dry")
    if target == "recorder":
        h["owl"] = h["recorder_detected"]
    elif target == "any":
        h["owl"] = ((h["camera_detected"] == 1) | (h["recorder_detected"] == 1)).astype(int)
    else:
        raise ValueError("target must be 'recorder' or 'any'")
    h.attrs["dropped_missing"] = n_all - len(h)
    cols = ["station", "river_system", "campaign", "datetime", "night", "hour", "month",
            "season", "breeding", "camera_detected", "recorder_detected", "owl"]
    return h[cols].reset_index(drop=True)


if __name__ == "__main__":
    h = load()
    print(f"{len(h)} station-hours with both sensors ({h.attrs['dropped_missing']} dropped for missing data)")
    print(h.groupby("river_system").agg(hours=("station", "size"), stations=("station", "nunique"),
                                        campaign=("campaign", "first")).to_string())
