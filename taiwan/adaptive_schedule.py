"""Per-station adaptive recorder schedule, evaluated month by month at every station.

Question: at each station, how many recorder-hours can we skip while still catching the owl calls
an always-on recorder caught (the ground truth in the data)?

How the schedule works (one station at a time, night by night, using only the past):
  1. For each hour of the night, the station keeps a running estimate of "how often does the
     recorder hear an owl in this hour here". It starts from the other river system's average for
     that hour (a weak starting guess, worth START_NIGHTS nights of data) and is updated only from
     hours this station actually recorded. Older nights count less (half weight after
     HALF_LIFE_DAYS), so the estimate follows the seasons.
  2. Record an hour if its estimate is at least THRESHOLD.
  3. Camera cue: if the camera saw an owl at this station earlier the same night, or on the
     previous sampled night, record the rest of that night.
  4. Audit: also record a random AUDIT_SHARE of the hours the schedule would skip, so the estimate
     can notice owls returning to hours it had switched off.

Baselines, per station-month:
  - always on:        100% of hours, 100% caught (today's cost)
  - random hours:     same number of hours as the schedule; catches the same share on average
  - skip edge hours:  fixed rule, record every hour except 17:00 and 06:00
  - oracle:           records only the hours with an owl (knows the future)

Outputs (outputs/taiwan/):
  adaptive_station_month.csv  one row per station-month: hours, owl-hours, % on and % caught for
                              the schedule and each baseline
  adaptive_station_month.png  one panel per station: % on vs % caught, by month
  adaptive_summary.csv        totals per station and for all stations
  adaptive_tradeoff.csv/.png  all stations together at several thresholds

Run: python -m taiwan.adaptive_schedule
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taiwan import config as C
from taiwan.load import load

THRESHOLD = 0.10          # record an hour if its estimated owl rate is at least this
THRESHOLDS_SWEEP = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45]
START_NIGHTS = 3          # strength of the starting guess, in nights of data
HALF_LIFE_DAYS = 30       # a night 30 days ago counts half as much as last night
AUDIT_SHARE = 0.10        # share of skipped hours recorded anyway, at random
SEEDS = range(10)         # the audit is random: results are averaged over these seeds
NIGHT_ORDER = list(range(17, 24)) + list(range(0, 7))


def prepare():
    h = load()
    h["night"] = pd.to_datetime(h["night"])
    h["ym"] = h["datetime"].dt.to_period("M")   # calendar month of the hour itself
    h["pos"] = h["hour"].map({x: i for i, x in enumerate(NIGHT_ORDER)})
    return h.sort_values(["station", "night", "pos"]).reset_index(drop=True)


def starting_guess(h, river):
    """Per-hour recorder detection rate at the other river system."""
    other = h[h.river_system != river]
    return other.groupby("hour")["recorder_detected"].mean().reindex(NIGHT_ORDER).fillna(0.05)


def run_station(st, prior, threshold, seed):
    """Simulate one station night by night. Returns a boolean 'recorded' array aligned with st."""
    rng = np.random.default_rng(seed)
    recorded = np.zeros(len(st), dtype=bool)
    obs_night, obs_hour, obs_y = [], [], []          # what this station has recorded so far
    prev_night_camera = False
    pos = 0
    for night, g in st.groupby("night", sort=True):
        now = np.datetime64(night, "ns")
        if obs_night:
            age = (now - np.array(obs_night, dtype="datetime64[ns]")) / np.timedelta64(1, "D")
            w = 0.5 ** (age / HALF_LIFE_DAYS)
            df = pd.DataFrame({"hour": obs_hour, "w": w, "wy": w * np.array(obs_y)})
            agg = df.groupby("hour")[["w", "wy"]].sum()
            seen_w, seen_wy = agg["w"].to_dict(), agg["wy"].to_dict()
        else:
            seen_w, seen_wy = {}, {}
        camera_cue = prev_night_camera
        for r in g.itertuples():
            rate = (START_NIGHTS * prior[r.hour] + seen_wy.get(r.hour, 0.0)) / (START_NIGHTS + seen_w.get(r.hour, 0.0))
            on = rate >= threshold or camera_cue or rng.random() < AUDIT_SHARE
            recorded[pos] = on
            pos += 1
            if on:
                obs_night.append(now), obs_hour.append(r.hour), obs_y.append(r.recorder_detected)
            if r.camera_detected:                    # the camera is always on: its sightings are free
                camera_cue = True
        prev_night_camera = bool(g["camera_detected"].any())
    return recorded


def simulate(h, threshold):
    """Share of seeds in which each station-hour was recorded (0..1)."""
    on = np.zeros(len(h))
    for river in h.river_system.unique():
        prior = starting_guess(h, river)
        for stn in h.loc[h.river_system == river, "station"].unique():
            idx = h.index[h.station == stn]
            on[idx] = np.mean([run_station(h.loc[idx], prior, threshold, s) for s in SEEDS], axis=0)
    return on


def pct(a, b):
    return np.where(b > 0, 100 * a / np.where(b > 0, b, 1), np.nan)


def tabulate(h, on, keys):
    rule = (~h["hour"].isin(C.EDGE_HOURS)).astype(float)
    d = h.assign(on=on, caught=on * h.recorder_detected, rule_on=rule, rule_caught=rule * h.recorder_detected)
    t = (d.groupby(keys)
           .agg(hours=("on", "size"), owl_hours=("recorder_detected", "sum"),
                hours_on=("on", "sum"), caught=("caught", "sum"),
                rule_hours_on=("rule_on", "sum"), rule_caught=("rule_caught", "sum"))
           .reset_index())
    t["missed"] = t.owl_hours - t.caught
    t["pct_on"] = pct(t.hours_on, t.hours)
    t["pct_caught"] = pct(t.caught, t.owl_hours)
    t["random_pct_caught"] = np.where(t.owl_hours > 0, t.pct_on, np.nan)
    t["rule_pct_on"] = pct(t.rule_hours_on, t.hours)
    t["rule_pct_caught"] = pct(t.rule_caught, t.owl_hours)
    t["oracle_pct_on"] = pct(t.owl_hours, t.hours)
    return t


def style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#e6e5e0")
    ax.set_axisbelow(True)


def plot_grid(t, path):
    stations = t[["river_system", "station"]].drop_duplicates().sort_values(["river_system", "station"])
    n, cols = len(stations), 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.8 * rows), dpi=130, sharey=True)
    axes = axes.ravel()
    for ax, (_, s) in zip(axes, stations.iterrows()):
        d = t[t.station == s.station].sort_values("ym")
        x = np.arange(len(d))
        ax.plot(x, d.pct_on, color="#8a8984", lw=2, marker="o", ms=4, label="% of hours recorder on")
        ax.plot(x, d.pct_caught, color="#2a78d6", lw=2, marker="o", ms=4, label="% of owl-hours caught")
        for xi, n_owl in enumerate(d.owl_hours):
            ax.text(xi, 104, f"{int(n_owl)}", ha="center", fontsize=6, color="#52514e")
        ax.set_xticks(x, [p.strftime("%b") for p in d.ym], fontsize=7)
        ax.set_title(f"{s.station} ({s.river_system}): {int(d.owl_hours.sum())} owl-hours, "
                     f"{int(round(d.caught.sum()))} caught", loc="left", fontsize=8.5)
        ax.set_ylim(0, 112)
        style(ax)
        ax.xaxis.grid(False)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(f"Every station, every month (schedule threshold {THRESHOLD}; "
                 f"small numbers above = owl-hours that month)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tradeoff(h, pts, path):
    rule = ~h["hour"].isin(C.EDGE_HOURS)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot([0, 100], [0, 100], color="#8a8984", ls=":", lw=2, label="Random hours")
    oracle_x = 100 * h.recorder_detected.mean()
    ax.plot([0, oracle_x, 100], [0, 100, 100], color="#52514e", ls="--", lw=1.5, label="Oracle")
    ax.plot(pts.pct_on, pts.pct_caught, color="#2a78d6", marker="o", lw=2, label="Adaptive schedule")
    for r in pts.itertuples():
        ax.annotate(f"{r.threshold:g}", (r.pct_on, r.pct_caught), textcoords="offset points",
                    xytext=(5, -11), fontsize=7, color="#52514e")
    ax.scatter([100 * rule.mean()], [100 * h.recorder_detected[rule].sum() / h.recorder_detected.sum()],
               color="#eb6834", s=50, zorder=4, label="Skip 17:00 and 06:00")
    ax.scatter([100], [100], color="#0b0b0b", s=40, zorder=4, label="Always on")
    ax.set_xlabel("Recorder on (% of hours, all 13 stations)")
    ax.set_ylabel("Owl-hours caught (% of always-on)")
    ax.set_title("Adaptive schedule at several thresholds (numbers on points)", loc="left", fontsize=10)
    ax.set_xlim(0, 102), ax.set_ylim(0, 102)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    h = prepare()

    on = simulate(h, THRESHOLD)
    t = tabulate(h, on, ["river_system", "station", "ym"])
    t.assign(ym=t.ym.astype(str)).round(1).to_csv(C.OUT / "adaptive_station_month.csv", index=False)
    plot_grid(t, C.OUT / "adaptive_station_month.png")

    s = tabulate(h, on, ["river_system", "station"])
    total = tabulate(h.assign(all="ALL"), on, ["all"]).rename(columns={"all": "station"}).assign(river_system="ALL")
    summ = pd.concat([s, total], ignore_index=True)
    summ.round(1).to_csv(C.OUT / "adaptive_summary.csv", index=False)

    pts = []
    for th in THRESHOLDS_SWEEP:
        o = on if th == THRESHOLD else simulate(h, th)
        pts.append({"threshold": th, "pct_on": 100 * o.mean(),
                    "pct_caught": 100 * (o * h.recorder_detected).sum() / h.recorder_detected.sum(),
                    "owl_hours_missed": (h.recorder_detected * (1 - o)).sum()})
    pts = pd.DataFrame(pts)
    pts.round({"pct_on": 1, "pct_caught": 1, "owl_hours_missed": 0}).to_csv(C.OUT / "adaptive_tradeoff.csv", index=False)
    plot_tradeoff(h, pts, C.OUT / "adaptive_tradeoff.png")

    pd.set_option("display.width", 200)
    print(summ[["river_system", "station", "hours", "owl_hours", "pct_on", "pct_caught", "missed",
                "rule_pct_on", "rule_pct_caught", "oracle_pct_on"]].round(1).to_string(index=False))
    print()
    print(pts.round({"pct_on": 1, "pct_caught": 1, "owl_hours_missed": 0}).to_string(index=False))


if __name__ == "__main__":
    main()
