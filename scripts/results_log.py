# Collects all M3 results into one table (results_log_m3.csv) and adds M1's rows (results_log.csv)
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUT, RESULTS, SEED
from plot_utils import AUDIO, CAMERA, THIRD, W, plt, save

COLS = ["experiment", "question", "metric", "value", "interval", "number of events", "shift level", "k", "tag"]
N_BOOT = 1000
PROJECT_DIR = Path(__file__).resolve().parent.parent
LEVEL = {"L0_same_sites": "seen site (source CV)", "L1_new_sites": "new site (source CV)",
         "L2_new_project": "new project (source CV)", "target_official": "new site (official target)",
         "drop L0->L1": "seen site -> new site", "drop L1->L2": "new site -> new project",
         "drop L0->L2": "seen site -> new project", "L0->L1": "seen site -> new site",
         "L1->L2": "new site -> new project", "L0->L2": "seen site -> new project"}
rows = []


def ci(lo, hi):
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


# One row in the team format; values on a 0-100 scale
def add(experiment, question, metric, value, interval, n, shift, k="0", tag="[Ours]"):
    rows.append(dict(zip(COLS, [experiment, question, metric, round(100 * float(value), 1), interval,
                                "" if n is None or pd.isna(n) else int(round(float(n))), shift, k, tag])))


def unit(metric, diff=False):
    name = {"accuracy": "accuracy", "macro_f1": "macro-F1", "overconfidence": "over-confidence",
            "ece_point_only": "ECE"}[metric]
    return f"{name} ({'points' if diff or metric in ('overconfidence', 'ece_point_only') else '%'})"


def read(name):
    p = RESULTS / name
    if not p.exists():
        print(f"missing {name} -> its rows are skipped")
        return None
    return pd.read_csv(p)


# Q1 shift levels
s = read("shift_results.csv")
if s is not None:
    for r in s.itertuples():
        drop = r.level.startswith("drop")
        sset = "clips scorable at every level" if r.clip_set == "common" else (
            "official target" if r.clip_set == "target" else "all scorable source clips")
        exp = (f"M3 {r.sensor} expert{' drop' if drop else ''} ({sset}, {r.n_sites} sites)")
        q = "Experts at new sites" if r.level == "target_official" else "Q1"
        add(exp, q, unit(r.metric, diff=drop), r.value, ci(r.ci_lo, r.ci_hi), r.n_clips, LEVEL[r.level])

# Q1 drop difference
d = read("shift_drop_diff.csv")
if d is not None:
    for r in d.itertuples():
        add(f"M3 audio drop - camera drop (H2; resampling {'sites' if r.unit == 'location_id' else 'projects'})",
            "Q1", unit(r.metric, diff=True), r.diff, ci(r.ci_lo, r.ci_hi), None, LEVEL[r.step])

# Shortcut tests
b = read("shortcut_results.csv")
if b is not None:
    for r in b.itertuples():
        within = r.test == "c1"
        exp = f"M3 shortcut test ({r.test}): {r.predictor}, {r.clip_set} ({r.n_sites} sites)"
        add(exp, "Shortcut test", unit(r.metric, diff=" - " in r.predictor), r.value, ci(r.ci_lo, r.ci_hi),
            r.n_clips, "new site within project (source CV)" if within else "new site (official target)")
for name, label in [("shortcut_night_results.csv", "project x light"),
                    ("shortcut_placement_results.csv", "project x light x placement")]:
    t = read(name)
    if t is None:
        continue
    for r in t.itertuples():
        add(f"M3 shortcut test (within {label}): {r.predictor}, {r.clip_set} ({r.n_sites} sites)", "Shortcut test",
            unit(r.metric, diff=" - " in r.predictor), r.value, ci(r.ci_lo, r.ci_hi), r.n_clips,
            "new site within group (source CV)")

# Adapted experts and selection headroom
a = read("adapted_results.csv")
if a is not None:
    a = a.rename(columns={"global": "glob"})
    for r in a.itertuples():
        k = str(int(r.k))
        if r.sensor == "oracle_headroom":
            add(f"M3 selection headroom, oracle - camera, adapted experts ({r.clip_set})", "Q3",
                "accuracy (points)", r.adapted, ci(r.adapted_lo, r.adapted_hi), None, "new site (official target)", k)
            add(f"M3 selection headroom, oracle - camera, global experts ({r.clip_set})", "Q3",
                "accuracy (points)", r.glob, "CI not stored (see the all-target-clips row)",
                None, "new site (official target)", k)
            add(f"M3 selection headroom change, adapted - global ({r.clip_set})", "Q3", "accuracy (points)",
                r.diff, ci(r.diff_lo, r.diff_hi), None, "new site (official target)", k)
            continue
        if r.clip_set in ("pool_minus_heldout", "same_minus_other_species"):
            add(f"M3 adapted {r.sensor}: gain contrast {r.clip_set}", "Q3", unit(r.metric, diff=True), r.diff,
                ci(r.diff_lo, r.diff_hi), None, "new site (official target)", k)
            continue
        base = f"M3 adapted {r.sensor} ({r.clip_set}, mean of 5 seeds)"
        if r.metric == "ece_point_only":
            add(base, "Q2/Q3", "ECE (points), adapted", r.adapted, "point only: bootstrap biased",
                r.n_clips_mean, "new site (official target)", k)
            continue
        add(base, "Q3", unit(r.metric), r.adapted, ci(r.adapted_lo, r.adapted_hi), r.n_clips_mean,
            "new site (official target)", k)
        add(f"M3 adapted - global {r.sensor} ({r.clip_set}, same clips)", "Q3", unit(r.metric, diff=True), r.diff,
            ci(r.diff_lo, r.diff_hi), r.n_clips_mean, "new site (official target)", k)

# Label-weight tuning
w = read("adapt_weight_tuning.csv")
if w is not None:
    g = w[w.w > 0].groupby(["sensor", "clip_set", "w"]).agg(gain=("gain", "mean"), n=("n_clips", "mean"),
                                                             chosen=("chosen", "max")).reset_index()
    for r in g.itertuples():
        add(f"M3 label-weight tuning on source pseudo-targets: {r.sensor} x{r.w:g}"
            f"{' (chosen)' if r.chosen else ''}, {r.clip_set}", "Q3 (design)", "accuracy gain over no labels (points)",
            r.gain, "no CI: mean of 3 source folds", r.n, "new site (source pseudo-target)", "3")

# Chance levels, calibration, oracle and who-is-right, recomputed from clip_events.csv
ev = pd.read_csv(OUT / "clip_events.csv")
t = ev[ev.split != "source"].reset_index(drop=True)
sites = t.location_id.astype(str).to_numpy()
uniq = np.unique(sites)
members = {u: np.flatnonzero(sites == u) for u in uniq}
rng = np.random.default_rng(SEED)
draws = [np.concatenate([members[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(N_BOOT)]
cc, ca = t.correct_camera.to_numpy(bool), t.correct_audio.to_numpy(bool)


def boot_add(exp, question, metric, f):
    v = np.array([f(r) for r in draws])
    add(exp, question, metric, f(np.arange(len(t))), ci(*np.percentile(v, [2.5, 97.5])), len(t),
        "new site (official target)")


n_sp = t.species.nunique()
add("M3 chance: uniform guess over species", "Experts at new sites", "accuracy (%)", 1 / n_sp, "exact", len(t),
    "new site (official target)")
maj = t.species.value_counts()
add(f"M3 chance: always the most common target species ({maj.index[0]})", "Experts at new sites", "accuracy (%)",
    maj.iloc[0] / len(t), "exact", len(t), "new site (official target)")
for sen in ["camera", "audio"]:
    conf, hit = t[f"conf_{sen}"].to_numpy(), t[f"correct_{sen}"].to_numpy(bool)
    boot_add(f"M3 {sen} expert calibration: mean confidence (global)", "Q2", "mean confidence (%)",
             lambda r: conf[r].mean())
    boot_add(f"M3 {sen} expert calibration: over-confidence (global)", "Q2", "over-confidence (points)",
             lambda r: conf[r].mean() - hit[r].mean())
boot_add("M3 oracle (camera or audio right), global experts", "Q3", "accuracy (%)", lambda r: (cc[r] | ca[r]).mean())
boot_add("M3 selection headroom, oracle - camera, global experts (all target clips)", "Q3", "accuracy (points)",
         lambda r: (cc[r] | ca[r]).mean() - cc[r].mean())
for name, m in [("both right", cc & ca), ("camera only right", cc & ~ca), ("audio only right", ~cc & ca),
                ("neither right", ~cc & ~ca)]:
    add(f"M3 who is right: {name} (global experts)", "Q3", "share of target clips (%)", m.mean(),
        "no CI: descriptive count", int(m.sum()), "new site (official target)")

# Accuracy by condition and per species
sl = read("results_slices.csv")
if sl is not None:
    for r in sl.itertuples():
        for sen, col in [("camera", "global_visual"), ("audio", "audio_only")]:
            add(f"M3 {sen} expert by condition: {r.slice}", "Conditions", "accuracy (%)", getattr(r, col),
                "no CI: directional only", r.n, "new site (official target)")
ps = read("results_per_species.csv")
if ps is not None:
    ps = ps.rename(columns={ps.columns[0]: "species"})
    for r in ps.itertuples():
        for sen, col in [("camera", "global_visual"), ("audio", "audio_only")]:
            add(f"M3 {sen} expert per species: {r.species}", "Experts at new sites", "F1 (%)", getattr(r, col),
                "no CI: directional only", r.n_test, "new site (official target)")

# Compute cost
for dev in ["cuda", "cpu"]:
    c = read(f"compute_summary_{dev}.csv")
    if c is None:
        continue
    for r in c.itertuples():
        is_ratio = "/" in r.quantity
        metric = r.quantity if is_ratio else (r.quantity.replace("_s", " (s per clip)")
                                              .replace("_gflop", " (GFLOP per clip)"))
        rows.append(dict(zip(COLS, [f"M3 compute cost on {dev}: {r.quantity}", "Compute cost (one-pager)",
                                    metric, round(float(r.mean), 4),
                                    f"[{r.mean_lo:.4g}, {r.mean_hi:.4g}] (bootstrap over clips)", int(r.n_clips),
                                    "n/a", "n/a", "[Ours]"])))

log = pd.DataFrame(rows, columns=COLS)
log.to_csv(RESULTS / "results_log_m3.csv", index=False)
print(f"M3 rows: {len(log)} -> {RESULTS / 'results_log_m3.csv'}")
print(log.groupby("question").size().to_string())

# Adds M1's station-level rows if available
m1 = None
zp, cp = PROJECT_DIR / "m1_station_results.zip", PROJECT_DIR / "results_log_rows.csv"
if cp.exists():
    m1 = pd.read_csv(cp)
elif zp.exists():
    with zipfile.ZipFile(zp) as z:
        name = next((n for n in z.namelist() if n.endswith("results_log_rows.csv")), None)
        if name:
            with z.open(name) as f:
                m1 = pd.read_csv(f)
if m1 is not None:
    if list(m1.columns) != COLS:
        print(f"M1 columns differ from the team format: {list(m1.columns)} -> combined file not written")
    else:
        both = pd.concat([m1, log], ignore_index=True)
        both.to_csv(RESULTS / "results_log.csv", index=False)
        print(f"combined with M1 ({len(m1)} rows) -> {RESULTS / 'results_log.csv'} ({len(both)} rows)")
else:
    print("M1 rows not found -> only results_log_m3.csv written")

# Fig 6: M1 owl coverage (needs M1's zip)
if zp.exists():
    with zipfile.ZipFile(zp) as z, z.open("coverage_overall.csv") as f:
        cov = pd.read_csv(f)
    scopes = [("all stations", "All 13 stations"), ("Kaoping", "Kaoping (5)"), ("Zengwen-Yanshui", "Zengwen-Yanshui (8)")]
    parts = [("recorder-only share of owl-hours", "Recorder only", AUDIO),
             ("both-sensor share of owl-hours", "Both", THIRD),
             ("camera-only share of owl-hours", "Camera only", CAMERA)]
    fig, ax = plt.subplots(figsize=(W, 1.9))
    yy = np.arange(len(scopes))[::-1]
    for yi, (sc, lab) in zip(yy, scopes):
        left = 0
        for metric, plab, col in parts:
            v = 100 * cov[(cov.scope == sc) & (cov.metric == metric)].value.iloc[0]
            ax.barh(yi, v, left=left, color=col, edgecolor="white", linewidth=1.5, height=0.65,
                    label=plab if yi == yy[0] else None)
            if v >= 6:
                ax.annotate(f"{v:.0f}%", (left + v / 2, yi), ha="center", va="center", fontsize=7, color="white")
            left += v
    ax.set_yticks(yy, [lab for _, lab in scopes])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of owl-hours (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3)
    ax.grid(axis="y", visible=False)
    save(fig, "fig6_m1_coverage")
