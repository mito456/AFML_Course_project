"""Count both / camera-only / recorder-only / neither, overall, by hour and by month.

Outputs (outputs/taiwan/):
  outcomes_overall.csv, outcomes_by_hour.csv, outcomes_by_month.csv
  outcomes_by_hour.png, outcomes_by_month.png

Plots show the share of station-hours with an owl, split by which sensor caught it.
"neither" hours are the remainder and are left out so the detections are readable.

Run: python -m taiwan.outcomes
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from taiwan import config as C
from taiwan.load import load

ORDER = ["both", "recorder_only", "camera_only", "neither"]
LABELS = {"both": "Both", "recorder_only": "Recorder only", "camera_only": "Camera only"}


def outcome(df):
    c, r = df["camera_detected"], df["recorder_detected"]
    return pd.Series(pd.NA, index=df.index).mask((c == 1) & (r == 1), "both") \
        .mask((c == 0) & (r == 1), "recorder_only").mask((c == 1) & (r == 0), "camera_only") \
        .mask((c == 0) & (r == 0), "neither")


def table(h, by):
    t = pd.crosstab(h[by], h["outcome"]).reindex(columns=ORDER, fill_value=0)
    t["station_hours"] = t[ORDER].sum(axis=1)
    return t


def plot(t, by, xlabels, title, path):
    share = t[["both", "recorder_only", "camera_only"]].div(t["station_hours"], axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    bottom = None
    x = range(len(share))
    for k in ["both", "recorder_only", "camera_only"]:
        ax.bar(x, share[k], bottom=bottom, color=C.COLOURS[k], label=LABELS[k],
               width=0.7, edgecolor="white", linewidth=1.5)
        bottom = share[k] if bottom is None else bottom + share[k]
    ax.set_xticks(list(x), xlabels)
    ax.set_ylabel("Station-hours with an owl (%)")
    ax.set_xlabel(by)
    ax.set_title(title, loc="left", fontsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#b5b4ae")
    ax.spines["bottom"].set_color("#b5b4ae")
    ax.yaxis.grid(True, color="#e6e5e0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    h = load()
    h["outcome"] = outcome(h)
    overall = h["outcome"].value_counts().reindex(ORDER)
    overall.to_csv(C.OUT / "outcomes_overall.csv", header=["station_hours"])

    # Hours in night order: 17:00 ... 23:00, 00:00 ... 06:00
    night_order = list(range(17, 24)) + list(range(0, 7))
    by_hour = table(h, "hour").reindex(night_order)
    by_month = table(h, "month").reindex(range(1, 13))
    by_hour.to_csv(C.OUT / "outcomes_by_hour.csv")
    by_month.to_csv(C.OUT / "outcomes_by_month.csv")

    plot(by_hour, "Hour of day (local)", [f"{x:02d}" for x in night_order],
         "Owl detections by hour, split by which sensor caught them", C.OUT / "outcomes_by_hour.png")
    plot(by_month, "Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
         "Owl detections by month, split by which sensor caught them", C.OUT / "outcomes_by_month.png")

    owl = overall[["both", "recorder_only", "camera_only"]].sum()
    print("station-hours:", overall.to_dict())
    print(f"recorder caught {(overall['both'] + overall['recorder_only']) / owl:.1%} of owl-hours; "
          f"camera alone added {overall['camera_only'] / owl:.1%}")
    br = h[h.outcome.isin(["both", "camera_only"])].groupby("breeding").size()
    print(f"camera detections at breeding stations: {br.get(1, 0)} of {br.sum()}")


if __name__ == "__main__":
    main()
