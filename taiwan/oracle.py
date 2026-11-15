"""Oracle: the fewest recorder-hours that still keep X% of owl detections.

The oracle knows in advance which hours hold an owl, so it records those first. It is the
upper bound any selector can reach; real selectors are compared against it.

Outputs (outputs/taiwan/): oracle.csv
Run: python -m taiwan.oracle [--target recorder|any]
"""
import argparse

import numpy as np
import pandas as pd

from taiwan import config as C
from taiwan.energy import baseline_days, battery_days, mwh_per_night
from taiwan.load import load

TARGETS = [0.5, 0.8, 0.9, 0.95, 1.0]


def oracle_fraction(h: pd.DataFrame, keep: float) -> float:
    """Share of recorder-hours needed to keep `keep` of owl-hours, or nan if unreachable.

    Only hours where the recorder detected an owl can be "kept" by recording; with
    target 'any', camera-only owl-hours stay missed whatever the recorder does.
    """
    reachable = int(((h.owl == 1) & (h.recorder_detected == 1)).sum())
    needed = int(np.ceil(keep * h.owl.sum()))
    if needed > reachable:
        return float("nan")
    return needed / len(h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=C.TARGET, choices=["recorder", "any"])
    a = p.parse_args()
    C.OUT.mkdir(parents=True, exist_ok=True)
    h = load(a.target)

    base_mwh = mwh_per_night(C.SAMPLED_HOURS_PER_NIGHT)
    rows = []
    for keep in TARGETS:
        f = oracle_fraction(h, keep)
        if np.isnan(f):
            rows.append({"keep_detections": keep, "recorder_hours_needed": np.nan})
            continue
        rows.append({
            "keep_detections": keep,
            "recorder_hours_needed": round(f, 4),
            "recorder_hours_skipped": round(1 - f, 4),
            "energy_saved": round(1 - mwh_per_night(f * C.SAMPLED_HOURS_PER_NIGHT) / base_mwh, 4),
            "battery_days": round(battery_days(f), 0),
        })
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / f"oracle_{a.target}.csv", index=False)

    print(f"target = {a.target}: {int(h.owl.sum())} owl-hours in {len(h)} station-hours")
    print(f"baseline schedule ({C.MINUTES_PER_SAMPLED_HOUR} min x {C.SAMPLED_HOURS_PER_NIGHT} h/night): "
          f"{baseline_days():.0f} battery days at {C.BATTERY_WH} Wh (assumed)")
    print(out.to_string(index=False))
    if a.target == "any":
        cap = ((h.owl == 1) & (h.recorder_detected == 1)).sum() / h.owl.sum()
        print(f"recorder alone can reach at most {cap:.1%} of owl-hours (camera-only hours are invisible to it)")


if __name__ == "__main__":
    main()
