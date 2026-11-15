"""Context-only recorder scheduler, tested on a river system it was not built on.

The camera is always on; the selector decides which hours to switch the recorder on, using
only hour of night, month and wet/dry season (no station or breeding features).

Three views, all written to outputs/taiwan/:

1. Full split  (selector_full_split.csv, selector_curve_<home>_to_<new>.png)
   Built on the whole home region, tested on the whole new region. Curve of owl-hours caught
   vs recorder-hours used, against an oracle, random hours, camera alone and the edge-hours rule.

2. Month by month  (selector_by_month.csv, selector_by_month_<new>.png)
   One deployment: learn on the first LEARNING_MONTHS months at the new place, freeze the
   schedule, then report every later month separately.

3. Every start month  (selector_every_start.csv, selector_every_start_summary.csv,
   selector_every_start.png)
   The same deployment repeated from every possible start month, so results show the spread
   (average, worst, best) rather than one lucky or unlucky start.

Two ways of building the schedule in views 2 and 3:
  - home only:  model and cutoff from the home region; ignores local data
  - home+local: model from home plus the learning months (weighted up to count as much as the
                home region); cutoff set on the learning months

Measures (both are reported; system-level is the headline):
  - system_det:   of all owl-hours (camera or recorder), share caught by camera + scheduled recorder
  - recorder_det: of hours the recorder heard an owl, share inside switched-on hours
  - camera_alone: system-level detection with the recorder never switched on
  - hours_used:   share of hours the recorder is switched on

Caveats for the slide: the Kaoping grouping of GR/JR/LG is inferred; region and year coincide
(accepted by the team); single study, two river systems.

Run: python -m taiwan.context_selector
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from taiwan import config as C
from taiwan.energy import mwh_per_night
from taiwan.load import load

FEATURES = ["hour", "month", "season"]


# --- data -----------------------------------------------------------------------

def prepare():
    h = load()
    h["ym"] = h["datetime"].dt.to_period("M")
    h["any_owl"] = ((h.camera_detected == 1) | (h.recorder_detected == 1)).astype(int)
    h["adds"] = ((h.recorder_detected == 1) & (h.camera_detected == 0)).astype(int)
    return h


ENC = None


def X(df):
    return ENC.transform(df[FEATURES].astype(str))


def fit(df, weights=None):
    return LogisticRegression(max_iter=2000).fit(X(df), df["adds"], sample_weight=weights)


def scores(model, df):
    return model.predict_proba(X(df))[:, 1]


def slug(s):
    return s.lower().replace("-", "_")


# --- schedule and measures ------------------------------------------------------

def cutoff_for_promise(model, cal):
    """Cutoff such that camera + recorder above the cutoff catch PROMISED_DETECTION of
    owl-hours in `cal`. Returns -inf if the camera alone already meets the promise."""
    cam = int(cal.camera_detected.sum() - ((cal.camera_detected == 1) & (cal.any_owl == 0)).sum())
    need = int(np.ceil(C.PROMISED_DETECTION * cal.any_owl.sum())) - cam
    if need <= 0:
        return np.inf  # nothing needs recording
    s = np.sort(scores(model, cal)[cal["adds"].to_numpy() == 1])[::-1]
    if need > len(s):
        return -np.inf  # promise unreachable: record everything
    return s[need - 1]


def measure(df, on):
    cam = df.camera_detected.to_numpy().astype(bool)
    rec = df.recorder_detected.to_numpy().astype(bool)
    owl = df.any_owl.to_numpy().astype(bool)
    caught = cam | (on & rec)
    return {
        "hours_used": on.mean(),
        "system_det": caught[owl].mean() if owl.any() else np.nan,
        "recorder_det": (on & rec).sum() / rec.sum() if rec.any() else np.nan,
        "camera_alone": cam[owl].mean() if owl.any() else np.nan,
        "owl_hours": int(owl.sum()),
    }


def energy_saved(fraction):
    return 1 - mwh_per_night(fraction * C.SAMPLED_HOURS_PER_NIGHT) / mwh_per_night(C.SAMPLED_HOURS_PER_NIGHT)


def build(method, home, learn):
    """Return (model, cutoff) for 'home only' or 'home+local'."""
    if method == "home only" or len(learn) == 0:
        m = fit(home)
        return m, cutoff_for_promise(m, home)
    w = np.r_[np.ones(len(home)), np.full(len(learn), len(home) / len(learn))]
    m = fit(pd.concat([home, learn]), w)
    cal = learn if learn["adds"].sum() >= C.MIN_LEARNING_POSITIVES else home
    return m, cutoff_for_promise(m, cal)


# --- view 1: full split curve ---------------------------------------------------

def curve(order_scores, df):
    """Recorder-hours used (x) vs system-level owl-hours caught (y), recording best-scored first."""
    o = np.argsort(-order_scores, kind="stable")
    cam = df.camera_detected.to_numpy()[o].astype(bool)
    rec = df.recorder_detected.to_numpy()[o].astype(bool)
    owl_total = df.any_owl.sum()
    gain = (rec & ~cam).astype(int)                     # owl-hours the recorder adds
    y = (cam.sum() + np.r_[0, np.cumsum(gain)]) / owl_total
    x = np.arange(len(df) + 1) / len(df)
    return x, y


def full_split(h, home_name, new_name):
    home, new = h[h.river_system == home_name], h[h.river_system == new_name]
    m, cut = build("home only", home, new.iloc[0:0])
    s = scores(m, new)
    rows = [{"home": home_name, "new_place": new_name, "policy": "context selector",
             **measure(new, s >= cut)}]
    rule_on = ~new["hour"].isin(C.EDGE_HOURS).to_numpy()
    rows.append({"home": home_name, "new_place": new_name, "policy": "skip edge hours", **measure(new, rule_on)})
    rows.append({"home": home_name, "new_place": new_name, "policy": "camera only",
                 **measure(new, np.zeros(len(new), bool))})
    rows.append({"home": home_name, "new_place": new_name, "policy": "always both",
                 **measure(new, np.ones(len(new), bool))})

    xm, ym = curve(s, new)
    xo, yo = curve(new["adds"].to_numpy().astype(float), new)
    rng = np.random.default_rng(0)
    xr, yr = curve(rng.random(len(new)), new)
    sel, rule, cam = rows[0], rows[1], rows[2]

    fig, ax = plt.subplots(figsize=(6.4, 5), dpi=150)
    ax.axhline(cam["system_det"] * 100, color=C.COLOURS["camera"], lw=1, ls="-.")
    ax.text(1, cam["system_det"] * 100 + 1.2, "camera alone", ha="left", fontsize=8, color="#52514e")
    ax.plot(xo * 100, yo * 100, color=C.COLOURS["oracle"], lw=2, ls="--", label="Oracle (upper bound)")
    ax.plot(xr * 100, yr * 100, color=C.COLOURS["random"], lw=2, ls=":", label="Random hours")
    ax.plot(xm * 100, ym * 100, color=C.COLOURS["model"], lw=2, label="Context selector (ranking)")
    ax.scatter([sel["hours_used"] * 100], [sel["system_det"] * 100], s=60, color=C.COLOURS["model"],
               edgecolor="white", linewidth=2, zorder=4,
               label=f"Selector at its cutoff (promised {C.PROMISED_DETECTION:.0%})")
    ax.scatter([rule["hours_used"] * 100], [rule["system_det"] * 100], s=60, color=C.COLOURS["rule"],
               edgecolor="white", linewidth=2, zorder=4,
               label=f"Skip {', '.join(f'{x:02d}:00' for x in C.EDGE_HOURS)}")
    ax.set_xlabel("Recorder switched on (% of hours)")
    ax.set_ylabel("Owl-hours caught, camera + recorder (%)")
    ax.set_title(f"Built on {home_name}, used at {new_name}", loc="left", fontsize=11)
    ax.set_xlim(0, 100), ax.set_ylim(0, 101)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#e6e5e0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(C.OUT / f"selector_curve_{slug(home_name)}_to_{slug(new_name)}.png")
    plt.close(fig)
    return rows


# --- views 2 and 3: deployments with a learning period -------------------------

def deployment(h, home_name, new_name, start_idx, method):
    home, new = h[h.river_system == home_name], h[h.river_system == new_name]
    months = sorted(new.ym.unique())
    learn = new[new.ym.isin(months[start_idx:start_idx + C.LEARNING_MONTHS])]
    test = new[new.ym.isin(months[start_idx + C.LEARNING_MONTHS:])]
    m, cut = build(method, home, learn)
    return learn, test, scores(m, test) >= cut


def by_month(h, home_name, new_name):
    rows = []
    for method in ["home only", "home+local"]:
        learn, test, on = deployment(h, home_name, new_name, 0, method)
        test = test.assign(on=on)
        for ym, g in test.groupby("ym"):
            rows.append({"home": home_name, "new_place": new_name, "method": method,
                         "learning_months": ", ".join(str(x) for x in sorted(learn.ym.unique())),
                         "month": str(ym), **measure(g, g["on"].to_numpy())})
    df = pd.DataFrame(rows)

    months = df["month"].unique()
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    for i, (method, key) in enumerate([("home only", "home_only"), ("home+local", "home_local")]):
        d = df[df.method == method].set_index("month").reindex(months)
        ax.bar(x + (i - 0.5) * 0.38, d["system_det"] * 100, width=0.36, color=C.COLOURS[key],
               label=method, edgecolor="white", linewidth=1.5)
    cam = df[df.method == "home only"].set_index("month").reindex(months)["camera_alone"] * 100
    ax.scatter(x, cam, marker="_", s=300, color="#0b0b0b", zorder=4, label="camera alone")
    ax.axhline(C.PROMISED_DETECTION * 100, color="#52514e", lw=1, ls="--")
    ax.text(len(months) - 0.5, C.PROMISED_DETECTION * 100 + 1.5, f"promised {C.PROMISED_DETECTION:.0%}",
            ha="right", fontsize=8, color="#52514e")
    ax.set_xticks(x, [pd.Period(mo).strftime("%b %y") for mo in months], fontsize=8)
    ax.set_ylabel("Owl-hours caught, camera + recorder (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{new_name}: schedule frozen after {C.LEARNING_MONTHS} learning months, each later month",
                 loc="left", fontsize=10)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#e6e5e0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(C.OUT / f"selector_by_month_{slug(new_name)}.png")
    plt.close(fig)
    return df


def every_start(h, pairs):
    rows = []
    for home_name, new_name in pairs:
        months = sorted(h[h.river_system == new_name].ym.unique())
        for s in range(0, len(months) - C.LEARNING_MONTHS - 1):
            for method in ["home only", "home+local"]:
                learn, test, on = deployment(h, home_name, new_name, s, method)
                if test.any_owl.sum() == 0:
                    continue
                rows.append({"home": home_name, "new_place": new_name, "method": method,
                             "start_month": str(months[s]), "learning_owl_hours": int(learn.any_owl.sum()),
                             "test_months": len(test.ym.unique()), **measure(test, on)})
    df = pd.DataFrame(rows)
    summary = (df.groupby(["new_place", "method"])
                 .agg(starts=("start_month", "size"),
                      hours_used=("hours_used", "mean"),
                      system_det_mean=("system_det", "mean"),
                      system_det_worst=("system_det", "min"),
                      system_det_best=("system_det", "max"),
                      recorder_det_mean=("recorder_det", "mean"),
                      camera_alone_mean=("camera_alone", "mean"))
                 .reset_index())
    summary["energy_saved"] = summary["hours_used"].map(energy_saved)

    fig, axes = plt.subplots(1, len(pairs), figsize=(10, 4.2), dpi=150, sharey=True)
    for ax, (home_name, new_name) in zip(np.atleast_1d(axes), pairs):
        d = df[df.new_place == new_name]
        for method, key in [("home only", "home_only"), ("home+local", "home_local")]:
            g = d[d.method == method]
            ax.plot(range(len(g)), g["system_det"] * 100, marker="o", ms=6, lw=2,
                    color=C.COLOURS[key], label=method, markeredgecolor="white", markeredgewidth=1.5)
        g = d[d.method == "home only"]
        ax.plot(range(len(g)), g["camera_alone"] * 100, lw=1, ls="-.", color=C.COLOURS["camera"],
                label="camera alone")
        ax.axhline(C.PROMISED_DETECTION * 100, color="#52514e", lw=1, ls="--")
        ax.set_xticks(range(len(g)), [pd.Period(x).strftime("%b %y") for x in g["start_month"]],
                      rotation=45, fontsize=8)
        ax.set_title(f"New place: {new_name}", loc="left", fontsize=10)
        ax.set_xlabel("Deployment start month")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.yaxis.grid(True, color="#e6e5e0")
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel("Owl-hours caught in later months (%)")
    np.atleast_1d(axes)[0].set_ylim(0, 105)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Same deployment from every start month ({C.LEARNING_MONTHS} learning months, "
                 f"promise {C.PROMISED_DETECTION:.0%}, dashed)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(C.OUT / "selector_every_start.png")
    plt.close(fig)
    return df, summary


def main():
    global ENC
    C.OUT.mkdir(parents=True, exist_ok=True)
    h = prepare()
    ENC = OneHotEncoder(handle_unknown="ignore").fit(h[FEATURES].astype(str))
    regions = sorted(h["river_system"].unique())
    pairs = [(a, b) for a in regions for b in regions if a != b]

    full = pd.DataFrame([r for hn, nn in pairs for r in full_split(h, hn, nn)])
    full["energy_saved"] = full["hours_used"].map(energy_saved)
    full.round(3).to_csv(C.OUT / "selector_full_split.csv", index=False)

    months = pd.concat([by_month(h, hn, nn) for hn, nn in pairs])
    months.round(3).to_csv(C.OUT / "selector_by_month.csv", index=False)

    starts, summary = every_start(h, pairs)
    starts.round(3).to_csv(C.OUT / "selector_every_start.csv", index=False)
    summary.round(3).to_csv(C.OUT / "selector_every_start_summary.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["new_place", "policy", "hours_used", "system_det", "recorder_det", "camera_alone", "energy_saved"]
    print("== 1. Full split ==")
    print(full[cols].round(2).to_string(index=False))
    print("\n== 2. Month by month (first deployment) ==")
    print(months[["new_place", "method", "month", "owl_hours", "hours_used", "system_det", "camera_alone"]]
          .round(2).to_string(index=False))
    print(f"\n== 3. Every start month (promise {C.PROMISED_DETECTION:.0%}) ==")
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
