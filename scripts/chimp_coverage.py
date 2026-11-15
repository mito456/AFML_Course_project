"""External check of the coverage pattern: chimpanzees, Issa Valley, Tanzania (Crunchant et al. 2020).

Data: Dryad doi:10.5061/dryad.5dv41ns34 (CC0), PresenceAbsence_matrices_CT_and_PAM.xlsx.
Design (from the paper, Methods): 12 acoustic sensors in 3 clusters of 4, rotated every two weeks
between 4 positions (12 cluster-positions); recording 30 min of every hour 06:00-19:30; calls detected
within ~500 m of a sensor; detections pooled per cluster; checked by hand in Raven.
53 motion-triggered video cameras (21 systematic, 32 targeted), running 24 h, detection range <= 29 m.

Pairing: a camera is co-located with an acoustic cluster-position if it lies within RADIUS_M of any of
that position's 4 sensors. Unit = site-day: one acoustic cluster-position on one date when it recorded
and at least one co-located camera was working. Acoustic day = chimpanzees heard in any half-hour file;
camera day = chimpanzees filmed by any co-located camera that day.

Run: python scripts/chimp_coverage.py
Outputs: outputs/chimp/chimp_site_days.csv, chimp_coverage.csv, chimp_pairs.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data/raw/chimp_issa2018/PresenceAbsence_matrices_CT_and_PAM.xlsx"
OUT = ROOT / "outputs/chimp"
RADII = [250, 500, 1000]   # 500 m = the paper's acoustic detection buffer (main result)
B = 2000


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load():
    x = pd.ExcelFile(XLSX)
    # Acoustic: rows = 12 cluster-positions ("Valley" 1-12), columns = (date, hour) half-hour files
    p = pd.read_excel(x, "PAM_PresenceAbsence", header=None)
    dates = pd.to_datetime(p.iloc[1, 2:]).dt.normalize().to_numpy()
    vals = p.iloc[2:, 2:].apply(pd.to_numeric, errors="coerce")
    vals.index = p.iloc[2:, 1].astype(int).to_numpy()          # site 1..12
    vals.columns = dates
    pam_day = vals.T.groupby(level=0).agg(lambda s: np.nan if s.isna().all() else float(s.max())).T
    pam_day.index.name = "site"

    meta = pd.read_excel(x, "PAM_Metadata").ffill()
    meta["site"] = ((meta.Cluster - 1) * 4 + meta.Valley).astype(int)   # cluster c, position v -> 1..12

    c = pd.read_excel(x, "CT_PresenceAbsence", header=None)
    cam = c.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")
    cam.index = c.iloc[1:, 0].astype(str).to_numpy()
    cam.columns = pd.to_datetime(c.iloc[0, 1:]).dt.normalize().to_numpy()
    cmeta = pd.read_excel(x, "CT_Metadata")
    return pam_day, meta, cam, cmeta


def pair(meta, cmeta):
    rows = []
    for site, g in meta.groupby("site"):
        for _, cr in cmeta.iterrows():
            d = haversine_m(g.Latitude.to_numpy(), g.Longitude.to_numpy(), cr.Latitude, cr.Longitude).min()
            rows.append(dict(site=site, camera=cr.Name, method=cr.Method, distance_m=round(d)))
    return pd.DataFrame(rows)


def site_days(pam_day, cam, pairs, radius):
    out = []
    for site, g in pairs[pairs.distance_m <= radius].groupby("site"):
        cams = [c for c in g.camera if c in cam.index]
        a = pam_day.loc[site]
        cm = cam.loc[cams]
        for day in a.index[a.notna()]:
            if day not in cm.columns:
                continue
            cday = cm[day]
            if cday.notna().sum() == 0:
                continue
            out.append(dict(site=site, date=day, cameras_in_zone=len(cams), cameras_working=int(cday.notna().sum()),
                            acoustic=int(a[day]), camera=int(cday.max())))
    d = pd.DataFrame(out)
    d["radius_m"] = radius
    return d


def boot_share(d, num_mask, den_mask):
    g = d.assign(n=num_mask.astype(int), k=den_mask.astype(int)).groupby("site")[["n", "k"]].sum()
    point = g.n.sum() / g.k.sum() if g.k.sum() else np.nan
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = g.n.to_numpy()[idx].sum(1) / g.k.to_numpy()[idx].sum(1)
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return point, lo, hi, len(g)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pam_day, meta, cam, cmeta = load()
    pairs = pair(meta, cmeta)
    pairs.to_csv(OUT / "chimp_pairs.csv", index=False)
    rows, all_days = [], []
    for r in RADII:
        d = site_days(pam_day, cam, pairs, r)
        all_days.append(d)
        both = (d.acoustic == 1) & (d.camera == 1)
        a_only = (d.acoustic == 1) & (d.camera == 0)
        c_only = (d.acoustic == 0) & (d.camera == 1)
        chimp = both | a_only | c_only
        row = dict(radius_m=r, sites=d.site.nunique(), cameras=int(pairs[pairs.distance_m <= r].camera.nunique()),
                   site_days=len(d), chimp_days=int(chimp.sum()), both=int(both.sum()),
                   acoustic_only=int(a_only.sum()), camera_only=int(c_only.sum()), neither=int((~chimp).sum()))
        for name, m in [("acoustic_only_share", a_only), ("camera_only_share", c_only), ("both_share", both)]:
            p, lo, hi, _ = boot_share(d, m, chimp)
            row[name], row[name + "_lo"], row[name + "_hi"] = p, lo, hi
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.concat(all_days).to_csv(OUT / "chimp_site_days.csv", index=False)
    res.round(4).to_csv(OUT / "chimp_coverage.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(pairs[pairs.distance_m <= 500].groupby("site").agg(cams=("camera", "nunique"),
          nearest_m=("distance_m", "min")).to_string())


if __name__ == "__main__":
    main()
