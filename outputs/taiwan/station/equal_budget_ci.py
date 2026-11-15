"""Intervals for the equal-budget comparison (Task 3).

For each target river system: per-station recorder-on hours and owl-hours caught for every adaptive
threshold (GRID, seeds 0-4, prior from the source system) and every fixed-schedule cutoff. Each of 2,000
station resamples rebuilds both pooled curves (% on vs % caught), reads % caught at 20/30/40/60% on by
linear interpolation (no extrapolation: a budget below a curve's lowest point is left blank), and takes
adaptive minus fixed. The adaptive curve bottoms out near 23-29% on (10% spot checks + camera tip-offs). Run: python -m taiwan.equal_budget_ci
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from taiwan import adaptive_schedule as A
from taiwan import station_results as S

BUDGETS = [20, 30, 40, 60]


def per_station(tgt, on):
    x = tgt.assign(on=on[tgt.index].to_numpy(), caught=on[tgt.index].to_numpy() * tgt.recorder_detected, one=1)
    return x.groupby("station")[["on", "one", "caught", "recorder_detected"]].sum()


def curve_tables(h, pool):
    out = {}
    for src_name in S.RIVERS:
        tgt_name = [r for r in S.RIVERS if r != src_name][0]
        src, tgt = h[h.river_system == src_name], h[h.river_system == tgt_name]
        stations = list(tgt.station.unique())
        prior = {s: S.hourly_mean(src) for s in stations}
        ad = [per_station(tgt, S.simulate(tgt, stations, prior, th, pool)) for th in S.GRID]
        cuts = sorted(set(src.groupby(["month", "hour"])["recorder_detected"].mean().round(4)), reverse=True)
        fx = [per_station(tgt, S.fixed_schedule_on(src, tgt, c)[0]) for c in cuts + [0.0]]
        out[tgt_name] = (stations, ad, fx)
    return out


def read_at(tables, w):
    """Pooled curve under station weights w (counts of each station in the resample); % caught at budgets."""
    on = np.array([(t.on.to_numpy() * w).sum() / (t.one.to_numpy() * w).sum() for t in tables]) * 100
    cg = np.array([(t.caught.to_numpy() * w).sum() / (t.recorder_detected.to_numpy() * w).sum() for t in tables]) * 100
    o = np.argsort(on)
    on, cg = on[o], np.maximum.accumulate(cg[o])
    return np.array([np.interp(b, on, cg, left=np.nan, right=np.nan) for b in BUDGETS])


def main():
    h = A.prepare()
    with ProcessPoolExecutor(max_workers=2) as pool:
        tabs = curve_tables(h, pool)
    rows = []
    rng = np.random.default_rng(0)
    for tgt_name, (stations, ad, fx) in tabs.items():
        n = len(stations)
        pa, pf = read_at(ad, np.ones(n)), read_at(fx, np.ones(n))
        bs = []
        for _ in range(S.B):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                a, f = read_at(ad, w), read_at(fx, w)
            bs.append(np.r_[a, f, a - f])
        bs = np.array(bs)
        lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
        k = len(BUDGETS)
        for i, b in enumerate(BUDGETS):
            rows.append(dict(target=tgt_name, budget_pct_on=b, stations=n,
                             owl_hours=int(sum(t.recorder_detected.sum() for t in ad[:1])),
                             fixed=pf[i], fixed_lo=lo[k + i], fixed_hi=hi[k + i],
                             adaptive=pa[i], adaptive_lo=lo[i], adaptive_hi=hi[i],
                             diff=pa[i] - pf[i], diff_lo=lo[2 * k + i], diff_hi=hi[2 * k + i],
                             resamples_reaching_budget=np.mean(~np.isnan(bs[:, 2 * k + i])),
                             share_diff_positive=np.mean(bs[~np.isnan(bs[:, 2 * k + i]), 2 * k + i] > 0)))
    res = pd.DataFrame(rows)
    res.round(2).to_csv(S.OUT / "equal_budget_ci.csv", index=False)
    print(res.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
