"""Clean the Auwahi (Maui) 2018 hoary bat thermal-video and acoustic detections.

Inputs : data/raw/auwahi_bats2018/{video_detections,acoustic_detections,nightly_summary}.csv
Outputs: data/processed/auwahi_video_events.csv
         data/processed/auwahi_acoustic_events.csv
         data/processed/auwahi_nights.csv
         data/processed/auwahi_event_matches.csv

Source: Gorresen, P.M. (2020), Maui Island, Hawaiian hoary bat behavior at wind
turbines, 2018. USGS data release, doi:10.5066/P937H9LQ, CC0 1.0.
Report: Gorresen, Cryan & Tredinnick (2020), USGS Index ID 70216926.

Sensors ran independently at 4 turbines (2, 4, 5, 7), Aug-Nov 2018: thermal video
cameras and acoustic detectors (microphones at the nacelle, per the report).

Conventions found in the source files:
- `Date` is the night's evening date, also for events after midnight (video events
  per Date match the nightly table exactly). Times are local (HST, UTC-10, no DST).
- Acoustic rows are individual call files grouped by `Event_ID`; IDs repeat across
  dates, so an event is (Turbine, Date, Event_ID). 495 files have Event_ID "na"
  (not assigned to an event) and are kept in the file table but not in events.
- Nightly table uses "na" for nights a sensor was not sampled.

Usage: python scripts/clean_auwahi.py [--tol SECONDS]
"""
import argparse
from pathlib import Path

import pandas as pd
from astral import LocationInfo
from astral.sun import sun

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/auwahi_bats2018"
OUT = ROOT / "data/processed"
TZ = "Pacific/Honolulu"
SITE_LAT, SITE_LON = 20.60, -156.33  # Auwahi Wind, approximate (not fitted)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tol", type=float, default=60.0,
                   help="seconds of slack when matching video and acoustic events")
    return p.parse_args()


def local_time(df):
    """Date is the night's date; times before noon belong to the next calendar day."""
    d = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    t = pd.to_timedelta(df["Time"])
    after_midnight = (t < pd.Timedelta(hours=12)).astype(int)
    ts = d + t + pd.to_timedelta(after_midnight, unit="D")
    return ts.dt.tz_localize(TZ), d.dt.date


def add_sun(df, time_col):
    obs = LocationInfo("auwahi", "US", "UTC", SITE_LAT, SITE_LON).observer
    sunsets = {n: pd.Timestamp(sun(obs, date=n)["sunset"]) for n in df["night"].unique()}
    ss = df["night"].map(sunsets)
    df["mins_since_sunset"] = ((df[time_col].dt.tz_convert("UTC") - ss).dt.total_seconds() / 60).round(1)
    df["hour_local"] = df[time_col].dt.hour
    return df


def main():
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Video: one row per event.
    v_src = pd.read_csv(RAW / "video_detections.csv")
    start, night = local_time(v_src)
    v = pd.DataFrame({
        "video_event_id": "V" + v_src["Event_ID"].astype(str).str.zfill(3),
        "turbine": v_src["Turbine"],
        "night": night,
        "start_local": start,
        "end_local": start + pd.to_timedelta(v_src["Duration"], unit="s"),
        "duration_s": v_src["Duration"],
        "video_file": v_src["File"],
        "rpm": v_src["RPM"],
        "flight_type": v_src["Flight_type"].str.strip().str.lower(),
        "bats_visible": v_src["Bats_visible"],
        "passes": v_src["Passes"],
        "chase": v_src["Chase"].str.strip().str.lower().map({"y": 1, "n": 0}),
        "nacelle_proximity": v_src["Nacelle_proximity"].str.strip().str.lower().map({"y": 1, "n": 0}),
    })
    v = add_sun(v, "start_local")
    assert v["video_event_id"].is_unique

    # Acoustic: call files grouped into events.
    f_src = pd.read_csv(RAW / "acoustic_detections.csv", dtype={"Event_ID": str})
    t, night = local_time(f_src)
    files = pd.DataFrame({"turbine": f_src["Turbine"], "night": night, "event_id_raw": f_src["Event_ID"],
                          "time_local": t, "file": f_src["File"], "terminal": f_src["Terminal"]})
    files = files[files["event_id_raw"].str.strip().str.lower() != "na"]
    ac = (files.groupby(["turbine", "night", "event_id_raw"])
               .agg(start_local=("time_local", "min"), end_local=("time_local", "max"),
                    n_files=("file", "size"), feeding_buzz=("terminal", "max"))
               .reset_index().sort_values(["turbine", "start_local"]))
    ac.insert(0, "acoustic_event_id", ["A" + str(i).zfill(3) for i in range(1, len(ac) + 1)])
    ac["duration_s"] = (ac["end_local"] - ac["start_local"]).dt.total_seconds()
    ac = add_sun(ac, "start_local")

    # Nights: both sensors' effort and counts, with weather and turbine context.
    n_src = pd.read_csv(RAW / "nightly_summary.csv", na_values="na")
    nights = pd.DataFrame({
        "turbine": n_src["Turbine"],
        "night": pd.to_datetime(n_src["Date"], format="%m/%d/%Y").dt.date,
        "video_hours": n_src["Visual_sample_duration"],
        "video_events": n_src["Visual_detections"],
        "acoustic_hours": n_src["Acoustic_sample_duration"],
        "acoustic_events": n_src["Acoustic_detections"],
        "rpm_mean": n_src["RPM_mean"],
        "precip_mm": n_src["Precipitation"],
        "wind_mean": n_src["Wind_mean"],
        "wind_sd": n_src["Wind_Std_Dev"],
        "rpm_starts": n_src["RPM_starts"],
    })
    nights["both_sampled"] = (nights["video_hours"].notna() & nights["acoustic_hours"].notna()).astype(int)

    # Match events on nights both sensors ran: overlap of [start - tol, end + tol].
    tol = pd.Timedelta(seconds=a.tol)
    both = set(map(tuple, nights.loc[nights.both_sampled == 1, ["turbine", "night"]].values))
    vb = v[[(x, y) in both for x, y in zip(v.turbine, v.night)]]
    ab = ac[[(x, y) in both for x, y in zip(ac.turbine, ac.night)]]
    pairs = []
    for r in vb.itertuples():
        c = ab[(ab.turbine == r.turbine) & (ab.start_local <= r.end_local + tol) & (ab.end_local >= r.start_local - tol)]
        pairs += [(r.video_event_id, x) for x in c.acoustic_event_id]
    m = pd.DataFrame(pairs, columns=["video_event_id", "acoustic_event_id"])
    seen_v, seen_a = set(m.video_event_id), set(m.acoustic_event_id)
    rows = [{"video_event_id": i, "acoustic_event_id": pd.NA, "outcome": "video_only"} for i in vb.video_event_id if i not in seen_v]
    rows += [{"video_event_id": pd.NA, "acoustic_event_id": i, "outcome": "acoustic_only"} for i in ab.acoustic_event_id if i not in seen_a]
    m["outcome"] = "both"
    matches = pd.concat([m, pd.DataFrame(rows)], ignore_index=True)
    meta = pd.concat([vb.set_index("video_event_id")[["turbine", "night", "start_local"]],
                      ab.set_index("acoustic_event_id")[["turbine", "night", "start_local"]]])
    key = matches["video_event_id"].fillna(matches["acoustic_event_id"])
    matches[["turbine", "night", "start_local"]] = meta.loc[key].values
    matches["tol_s"] = a.tol

    for df, name in [(v, "auwahi_video_events"), (ac, "auwahi_acoustic_events"),
                     (nights, "auwahi_nights"), (matches, "auwahi_event_matches")]:
        df.to_csv(OUT / f"{name}.csv", index=False)

    # Checks against the USGS report (384 video events, 244 acoustic events).
    assert len(v) == 384 and len(ac) == 244, (len(v), len(ac))
    print(f"video events {len(v)}, acoustic events {len(ac)}, nights {len(nights)} ({nights.both_sampled.sum()} with both sensors)")
    print(f"on both-sensor nights: video {len(vb)}, acoustic {len(ab)}; match tolerance {a.tol:g}s")
    print("outcomes:", matches.outcome.value_counts().to_dict())
    print(f"video events also heard: {len(seen_v)}/{len(vb)} ({len(seen_v)/len(vb):.0%}); "
          f"acoustic events also seen: {len(seen_a)}/{len(ab)} ({len(seen_a)/len(ab):.0%})")


if __name__ == "__main__":
    main()
