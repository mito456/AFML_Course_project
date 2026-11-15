"""Member 1, station level: coverage counts, policy transfer between river systems, conditions.

Writes outputs/taiwan/station/ (CSVs, two figures, results_log_rows.csv, station_results.md).
Hypotheses were written before this was run: outputs/taiwan/station/hypotheses_m1.md.

Pre-set choices
  - Target: catch 80% of recorder owl-hours (the hours an always-on recorder heard an owl).
  - Settings are tuned on the SOURCE river system only, then frozen and applied to the TARGET.
  - Seeds 0-4 for the random spot checks. 95% intervals by resampling stations (2,000 resamples).

Policies compared at the target river system
  - fixed schedule (no adaptation): record an hour if the source system's recorder detection rate for
    that month and hour of night is at least a cutoff; the cutoff is the one that reaches 80% on the source.
  - adaptive (current method): per-station weighted average updated nightly, starting from the source
    system's hourly average; threshold = the largest that reaches 80% on the source (where each source
    station starts from the other source stations' average, so no target data is used).
  - adaptive without camera tip-off: same, with the camera cue switched off.
  - random hours at the same budget (expected), and the oracle (records only owl-hours).

Run: python -m taiwan.station_results     (about 10 minutes on 2 cores)
"""
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taiwan import adaptive_schedule as A
from taiwan import config as C

OUT = C.OUT / "station"
TARGET = 0.80
SEEDS = range(5)
GRID = [0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40]
B = 2000
RIVERS = ["Zengwen-Yanshui", "Kaoping"]


# ---------------------------------------------------------------- helpers
def boot_ratio(df, num, den, group="station"):
    """Ratio of sums with a 95% interval from resampling stations."""
    g = df.groupby(group)[[num, den]].sum()
    point = g[num].sum() / g[den].sum() if g[den].sum() else np.nan
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    n, d = g[num].to_numpy()[idx].sum(1), g[den].to_numpy()[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = n / d
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return point, lo, hi, len(g)


def fmt(p, lo, hi, pct=True):
    k = 100 if pct else 1
    return f"{k*p:.1f} [{k*lo:.1f}, {k*hi:.1f}]"


def one_station(args):
    st, prior, th, seeds = args
    return np.mean([A.run_station(st, prior, th, s) for s in seeds], axis=0)


def simulate(data, stations, priors, th, pool):
    """Average recorded share per row for the given stations; priors: station -> Series by hour."""
    jobs = [(data[data.station == s], priors[s], th, SEEDS) for s in stations]
    on = pd.Series(0.0, index=data.index)
    for s, res in zip(stations, pool.map(one_station, jobs)):
        on[data.index[data.station == s]] = res
    return on


def hourly_mean(df):
    return df.groupby("hour")["recorder_detected"].mean().reindex(A.NIGHT_ORDER).fillna(0.05)


def caught_share(df, on):
    y = df.recorder_detected
    return (on[df.index] * y).sum() / y.sum()


# ---------------------------------------------------------------- task 1 + 4: coverage and conditions
def coverage(h):
    d = h.assign(
        both=((h.camera_detected == 1) & (h.recorder_detected == 1)).astype(int),
        rec_only=((h.camera_detected == 0) & (h.recorder_detected == 1)).astype(int),
        cam_only=((h.camera_detected == 1) & (h.recorder_detected == 0)).astype(int),
        neither=((h.camera_detected == 0) & (h.recorder_detected == 0)).astype(int),
    )
    d["owl"] = d.both + d.rec_only + d.cam_only
    d["breeding_station"] = np.where(d.breeding == 1, "breeding", "non-breeding")
    d["night_part"] = pd.cut(d.pos, [-1, 3, 7, 13], labels=["17:00-20:59", "21:00-00:59", "01:00-06:59"])
    return d


def coverage_tables(d):
    rows = []
    scopes = [("all stations", d)] + [(r, d[d.river_system == r]) for r in RIVERS]
    for name, x in scopes:
        for metric, num in [("recorder-only share of owl-hours", "rec_only"),
                            ("camera-only share of owl-hours", "cam_only"),
                            ("both-sensor share of owl-hours", "both")]:
            p, lo, hi, n = boot_ratio(x, num, "owl")
            rows.append(dict(scope=name, metric=metric, value=p, lo=lo, hi=hi, stations=n,
                             owl_hours=int(x.owl.sum()), count=int(x[num].sum())))
        g = x.groupby("station")[["rec_only", "cam_only"]].sum()
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(g), size=(B, len(g)))
        ratio = g.rec_only.to_numpy()[idx].sum(1) / np.maximum(g.cam_only.to_numpy()[idx].sum(1), 1)
        rows.append(dict(scope=name, metric="recorder-only / camera-only owl-hours",
                         value=g.rec_only.sum() / max(g.cam_only.sum(), 1),
                         lo=np.percentile(ratio, 2.5), hi=np.percentile(ratio, 97.5), stations=len(g),
                         owl_hours=int(x.owl.sum()), count=int(g.rec_only.sum())))
    overall = pd.DataFrame(rows)

    by_station = (d.groupby(["river_system", "station", "breeding_station"])
                    [["both", "rec_only", "cam_only", "neither", "owl"]].sum().reset_index())
    by_station["recorder_only_share"] = by_station.rec_only / by_station.owl.where(by_station.owl > 0)

    def cond(col, order=None):
        out = []
        for k, x in d.groupby(col, observed=True):
            p, lo, hi, n = boot_ratio(x, "rec_only", "owl")
            q, qlo, qhi, _ = boot_ratio(x.assign(one=1), "rec_only", "one")
            out.append(dict(condition=col, level=str(k), stations=n, station_hours=len(x),
                            owl_hours=int(x.owl.sum()), recorder_only=int(x.rec_only.sum()),
                            camera_only=int(x.cam_only.sum()),
                            recorder_only_share=p, share_lo=lo, share_hi=hi,
                            recorder_only_per_100h=100 * q, per100_lo=100 * qlo, per100_hi=100 * qhi))
        t = pd.DataFrame(out)
        if order is not None:
            t["_o"] = t.level.map({str(v): i for i, v in enumerate(order)})
            t = t.sort_values("_o").drop(columns="_o")
        return t

    conditions = pd.concat([
        cond("hour", A.NIGHT_ORDER),
        cond("night_part", ["17:00-20:59", "21:00-00:59", "01:00-06:59"]),
        cond("breeding_station", ["breeding", "non-breeding"]),
        cond("season", ["dry", "wet"]),
        cond("river_system", RIVERS),
    ], ignore_index=True)
    return overall, by_station, conditions


# ---------------------------------------------------------------- task 3: policy transfer
def fixed_schedule_on(src, tgt, cutoff=None):
    """Hour-by-month rates from the source; record target hours whose source rate >= cutoff."""
    cell = src.groupby(["month", "hour"])["recorder_detected"].mean()
    hour = src.groupby("hour")["recorder_detected"].mean()
    def rates(df):
        r = pd.Series([cell.get((m, hr), np.nan) for m, hr in zip(df.month, df.hour)], index=df.index)
        return r.fillna(df.hour.map(hour)).fillna(0)
    if cutoff is None:  # tune on source: smallest set of source hours reaching the target
        r = rates(src).sort_values(ascending=False, kind="stable")
        cum = src.loc[r.index, "recorder_detected"].cumsum() / src.recorder_detected.sum()
        cutoff = r.iloc[int(np.searchsorted(cum.to_numpy(), TARGET))]
    return (rates(tgt) >= cutoff).astype(float), cutoff


def tune_adaptive(src, data, pool):
    """Largest threshold reaching the target on the source, each source station starting from the
    other source stations' hourly average (leave-one-station-out)."""
    stations = list(src.station.unique())
    priors = {s: hourly_mean(src[src.station != s]) for s in stations}
    curve = []
    for th in GRID:
        on = simulate(data[data.river_system == src.river_system.iloc[0]], stations, priors, th, pool)
        curve.append((th, caught_share(src, on), on[src.index].mean()))
    ok = [c for c in curve if c[1] >= TARGET]
    return (max(ok)[0] if ok else min(GRID)), curve


def station_metrics(tgt, on):
    x = tgt.assign(on=on[tgt.index].to_numpy(), caught=on[tgt.index].to_numpy() * tgt.recorder_detected, one=1)
    p_on, on_lo, on_hi, n = boot_ratio(x, "on", "one")
    p_c, c_lo, c_hi, _ = boot_ratio(x, "caught", "recorder_detected")
    return dict(pct_on=100 * p_on, on_lo=100 * on_lo, on_hi=100 * on_hi,
                pct_caught=100 * p_c, caught_lo=100 * c_lo, caught_hi=100 * c_hi, stations=n,
                owl_hours=int(tgt.recorder_detected.sum()), owl_hours_caught=float(x.caught.sum()),
                recorder_hours=float(x.on.sum()), station_hours=len(tgt))


def transfer(h, pool):
    rows, curves = [], []
    no_cue = h.assign(camera_detected=0)
    for src_name in RIVERS:
        tgt_name = [r for r in RIVERS if r != src_name][0]
        src, tgt = h[h.river_system == src_name], h[h.river_system == tgt_name]
        tgt_stations = list(tgt.station.unique())
        prior_from_src = {s: hourly_mean(src) for s in tgt_stations}

        fixed_on, cutoff = fixed_schedule_on(src, tgt)
        rows.append(dict(source=src_name, target=tgt_name, policy="fixed schedule (no adaptation)",
                         setting=f"cutoff {cutoff:.3f}", **station_metrics(tgt, fixed_on)))

        for label, data in [("adaptive (borrowed prior + nightly updating)", h),
                            ("adaptive, no camera tip-off", no_cue)]:
            th, tune_curve = tune_adaptive(data[data.river_system == src_name], data, pool)
            on = simulate(data[data.river_system == tgt_name], tgt_stations, prior_from_src, th, pool)
            rows.append(dict(source=src_name, target=tgt_name, policy=label, setting=f"threshold {th}",
                             **station_metrics(tgt, on)))
            for t, c, o in tune_curve:
                curves.append(dict(source=src_name, target=src_name, policy=label + " (tuning on source)",
                                   threshold=t, pct_on=100 * o, pct_caught=100 * c))
            if label.startswith("adaptive (borrowed"):
                for t in GRID:
                    o = on if t == th else simulate(data[data.river_system == tgt_name], tgt_stations,
                                                    prior_from_src, t, pool)
                    curves.append(dict(source=src_name, target=tgt_name, policy=label, threshold=t,
                                       pct_on=100 * o[tgt.index].mean(), pct_caught=100 * caught_share(tgt, o)))
        # fixed-schedule curve on the target across cutoffs
        cell_rates = sorted(set(src.groupby(["month", "hour"])["recorder_detected"].mean().round(4)), reverse=True)
        for cut in cell_rates[:: max(1, len(cell_rates) // 20)] + [0.0]:
            o, _ = fixed_schedule_on(src, tgt, cut)
            curves.append(dict(source=src_name, target=tgt_name, policy="fixed schedule (no adaptation)",
                               threshold=cut, pct_on=100 * o.mean(), pct_caught=100 * caught_share(tgt, o)))

        p_owl = tgt.recorder_detected.mean()
        rows.append(dict(source=src_name, target=tgt_name, policy="oracle", setting="knows owl-hours",
                         pct_on=100 * p_owl, on_lo=np.nan, on_hi=np.nan, pct_caught=100.0, caught_lo=100.0,
                         caught_hi=100.0, stations=len(tgt_stations), owl_hours=int(tgt.recorder_detected.sum()),
                         owl_hours_caught=float(tgt.recorder_detected.sum()),
                         recorder_hours=float(tgt.recorder_detected.sum()), station_hours=len(tgt)))
    res = pd.DataFrame(rows)
    adaptive = res[res.policy.str.startswith("adaptive (borrowed")]
    for _, r in adaptive.iterrows():
        res.loc[len(res)] = dict(source=r.source, target=r.target, policy="random hours, same budget as adaptive",
                                 setting="expected value", pct_on=r.pct_on, on_lo=r.on_lo, on_hi=r.on_hi,
                                 pct_caught=r.pct_on, caught_lo=np.nan, caught_hi=np.nan, stations=r.stations,
                                 owl_hours=r.owl_hours, owl_hours_caught=r.owl_hours * r.pct_on / 100,
                                 recorder_hours=r.recorder_hours, station_hours=r.station_hours)
    order = ["fixed schedule (no adaptation)", "adaptive (borrowed prior + nightly updating)",
             "adaptive, no camera tip-off", "random hours, same budget as adaptive", "oracle"]
    res["_o"] = res.policy.map({p: i for i, p in enumerate(order)})
    return res.sort_values(["target", "_o"]).drop(columns="_o"), pd.DataFrame(curves)


# ---------------------------------------------------------------- figures
def style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#e6e5e0")
    ax.set_axisbelow(True)


def plot_transfer(res, curves, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, sharey=True)
    for ax, tgt in zip(axes, RIVERS):
        src = [r for r in RIVERS if r != tgt][0]
        cv = curves[(curves.target == tgt)]
        ax.plot([0, 100], [0, 100], color="#8a8984", ls=":", lw=2, label="Random hours")
        orc = res[(res.target == tgt) & (res.policy == "oracle")].iloc[0]
        ax.plot([0, orc.pct_on, 100], [0, 100, 100], color="#52514e", ls="--", lw=1.5, label="Oracle")
        f = cv[cv.policy == "fixed schedule (no adaptation)"].sort_values("pct_on")
        ax.plot(f.pct_on, f.pct_caught, color="#eb6834", lw=2, label="Fixed schedule from source")
        a = cv[cv.policy == "adaptive (borrowed prior + nightly updating)"].sort_values("pct_on")
        ax.plot(a.pct_on, a.pct_caught, color="#2a78d6", lw=2, marker="o", ms=3,
                label="Adaptive (borrowed prior + nightly updating)")
        for pol, col in [("fixed schedule (no adaptation)", "#eb6834"),
                         ("adaptive (borrowed prior + nightly updating)", "#2a78d6")]:
            r = res[(res.target == tgt) & (res.policy == pol)].iloc[0]
            ax.errorbar(r.pct_on, r.pct_caught, xerr=[[r.pct_on - r.on_lo], [r.on_hi - r.pct_on]],
                        yerr=[[r.pct_caught - r.caught_lo], [r.caught_hi - r.pct_caught]], fmt="o", ms=8,
                        color=col, mec="white", mew=1.5, capsize=3, zorder=5)
        ax.axhline(100 * TARGET, color="#52514e", lw=1, ls="-.")
        ax.text(99, 100 * TARGET - 4, "target 80%", ha="right", fontsize=8, color="#52514e")
        ax.set_title(f"Tuned on {src}, used at {tgt}", loc="left", fontsize=10)
        ax.set_xlabel("Recorder on (% of station-hours)")
        ax.set_xlim(0, 102), ax.set_ylim(0, 102)
        style(ax)
    axes[0].set_ylabel("Recorder owl-hours caught (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Policy transfer between river systems (big dots: the frozen setting, 95% intervals by station)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_conditions(cond, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, gridspec_kw={"width_ratios": [2.2, 1]})
    hr = cond[cond.condition == "hour"]
    x = np.arange(len(hr))
    axes[0].bar(x, hr.recorder_only_per_100h, color="#1baf7a", width=0.65,
                yerr=[hr.recorder_only_per_100h - hr.per100_lo, hr.per100_hi - hr.recorder_only_per_100h],
                capsize=2, ecolor="#52514e")
    axes[0].set_xticks(x, [f"{int(v):02d}" for v in hr.level], fontsize=8)
    axes[0].set_xlabel("Hour of night")
    axes[0].set_ylabel("Recorder-only owl-hours per 100 station-hours")
    axes[0].set_title("What the camera misses, by hour", loc="left", fontsize=10)
    br = cond[cond.condition == "breeding_station"]
    axes[1].bar(range(len(br)), br.recorder_only_per_100h, color="#1baf7a", width=0.6,
                yerr=[br.recorder_only_per_100h - br.per100_lo, br.per100_hi - br.recorder_only_per_100h],
                capsize=3, ecolor="#52514e")
    axes[1].set_xticks(range(len(br)), [f"{l}\n({s} stations)" for l, s in zip(br.level, br.stations)], fontsize=8)
    axes[1].set_title("…and by station type", loc="left", fontsize=10)
    for ax in axes:
        style(ax)
        ax.xaxis.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- results log + markdown
def log_rows(overall, cond, res):
    rows = []
    for _, r in overall[overall.scope == "all stations"].iterrows():
        pct = "share" in r.metric
        rows.append(dict(experiment="M1 coverage counts (Taiwan Grass-Owl)", question="Motivation: is the camera the bottleneck",
                         metric=r.metric, value=round(100 * r.value if pct else r.value, 1),
                         interval=f"[{100*r.lo if pct else r.lo:.1f}, {100*r.hi if pct else r.hi:.1f}]",
                         **{"number of events": r.owl_hours}, **{"shift level": "none"}, k="n/a", tag="[Ours]"))
    for _, r in res[res.policy != "random hours, same budget as adaptive"].iterrows():
        for metric, v, lo, hi in [("recorder owl-hours caught (%)", r.pct_caught, r.caught_lo, r.caught_hi),
                                  ("recorder on (% of station-hours)", r.pct_on, r.on_lo, r.on_hi)]:
            rows.append(dict(experiment=f"M1 policy transfer: {r.policy}", question="Q2 (station level)",
                             metric=f"{metric} at {r.target}, tuned on {r.source}", value=round(v, 1),
                             interval="" if np.isnan(lo) else f"[{lo:.1f}, {hi:.1f}]",
                             **{"number of events": r.owl_hours}, **{"shift level": "new river system"},
                             k="0 labelled; learns online from its own recordings", tag="[Ours]"))
    for cname in ["breeding_station", "night_part", "season"]:
        for _, r in cond[cond.condition == cname].iterrows():
            rows.append(dict(experiment=f"M1 conditions: {cname}", question="Conditions (where the recorder adds most)",
                             metric=f"recorder-only owl-hours per 100 station-hours, {r.level}",
                             value=round(r.recorder_only_per_100h, 2), interval=f"[{r.per100_lo:.2f}, {r.per100_hi:.2f}]",
                             **{"number of events": r.recorder_only}, **{"shift level": "none"}, k="n/a", tag="[Ours]"))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = A.prepare()
    d = coverage(h)
    overall, by_station, cond = coverage_tables(d)
    overall.round(4).to_csv(OUT / "coverage_overall.csv", index=False)
    by_station.round(3).to_csv(OUT / "coverage_by_station.csv", index=False)
    cond.round(4).to_csv(OUT / "conditions.csv", index=False)
    plot_conditions(cond, OUT / "conditions.png")

    with ProcessPoolExecutor(max_workers=2) as pool:
        res, curves = transfer(h, pool)
    res.round(2).to_csv(OUT / "transfer_results.csv", index=False)
    curves.round(2).to_csv(OUT / "transfer_curves.csv", index=False)
    plot_transfer(res, curves, OUT / "transfer.png")

    log_rows(overall, cond, res).to_csv(OUT / "results_log_rows.csv", index=False)

    pd.set_option("display.width", 220)
    print(overall.round(3).to_string(index=False))
    print(by_station.to_string(index=False))
    print(cond.round(2).to_string(index=False))
    print(res[["source", "target", "policy", "setting", "pct_on", "on_lo", "on_hi", "pct_caught",
               "caught_lo", "caught_hi", "stations", "owl_hours", "owl_hours_caught", "recorder_hours"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
