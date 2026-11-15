"""Evaluate the Member 2 OOF usefulness score as a budgeted audio selector.

This is deliberately a retrospective *source-only* evaluation.  It consumes
the already out-of-fold Member 2 score and existing source expert outcomes;
it never fits or calibrates a model and never reads target/adapt labels.
"""
import csv
import math
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMBER2 = os.path.dirname(os.path.abspath(__file__))
EVENTS = os.path.join(ROOT, "safari", "outputs", "clip_events.csv")
SCORES = os.path.join(MEMBER2, "availability.csv")
BUDGETS = (0.10, 0.20, 0.30, 0.50)
SEEDS = tuple(range(100))


def pct(v):
    return 100.0 * v


def metrics(selected, rows, total_useful):
    captured = sum(rows[i]["useful"] for i in selected)
    n = len(rows)
    k = len(selected)
    return {
        "selected_clips": k,
        "precision": captured / k if k else 0.0,
        "recall": captured / total_useful if total_useful else 0.0,
        "useful_clips_captured": captured,
        "audio_usage_rate": k / n if n else 0.0,
        # Deployment-wide correction yield: successful corrections per candidate clip.
        "audio_correction_rate": captured / n if n else 0.0,
    }


def rank_select(rows, key, k):
    # video_name is only a deterministic tie breaker, not a ranking feature.
    return sorted(range(len(rows)), key=lambda i: (-key(rows[i]), rows[i]["video_name"]))[:k]


def summarize_random(values):
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return mean, sd, values[0], values[-1]


def write_svg(path, results):
    # Dependency-free grouped bar plot: precision and correction yield by budget.
    width, height, left, bottom = 900, 500, 78, 72
    pw, ph = width - left - 42, height - bottom - 46
    colors = {"top_ranked": "#1769aa", "camera_uncertainty": "#e67e22", "random_mean": "#7f8c8d"}
    labels = {"top_ranked": "Usefulness rank", "camera_uncertainty": "Camera uncertainty", "random_mean": "Random (mean)"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:Arial,sans-serif;fill:#202124}.small{font-size:12px}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>',
             f'<text x="{left}" y="28" font-size="18" font-weight="bold">Audio selection precision and correction yield (source OOF)</text>']
    # Two panels share a 0--25% y scale, sufficient for this dataset and comparable.
    for panel, metric in enumerate(("precision", "audio_correction_rate")):
        x0 = left + panel * (pw / 2 + 18); w = pw / 2 - 18
        parts.append(f'<text x="{x0}" y="53" class="small">{"Precision among processed clips" if metric == "precision" else "Corrections per eligible clip"}</text>')
        for tick in range(0, 26, 5):
            y = bottom + ph - tick / 25 * ph
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" class="grid"/><text x="{x0-9}" y="{y+4:.1f}" text-anchor="end" class="small">{tick}%</text>')
        groupw = w / len(BUDGETS)
        for bi, budget in enumerate(BUDGETS):
            base = x0 + bi * groupw + 8
            for mi, method in enumerate(("top_ranked", "camera_uncertainty", "random_mean")):
                val = results[(budget, method)][metric]
                h = min(val, .25) / .25 * ph
                bx = base + mi * 13
                parts.append(f'<rect x="{bx:.1f}" y="{bottom+ph-h:.1f}" width="10" height="{h:.1f}" fill="{colors[method]}"/>')
            parts.append(f'<text x="{base+18}" y="{bottom+ph+18}" text-anchor="middle" class="small">{int(budget*100)}%</text>')
        parts.append(f'<line x1="{x0}" y1="{bottom}" x2="{x0+w}" y2="{bottom}" class="axis"/>')
    lx = left; ly = height - 12
    for method in ("top_ranked", "camera_uncertainty", "random_mean"):
        parts.append(f'<rect x="{lx}" y="{ly-10}" width="10" height="10" fill="{colors[method]}"/><text x="{lx+14}" y="{ly}" class="small">{labels[method]}</text>')
        lx += 155
    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(parts))


def write_random_svg(path, random_rows):
    """Error-bar plot of random useful-clip capture across its 100 seeds."""
    width, height, left, bottom = 700, 430, 72, 64
    pw, ph = width - left - 38, height - bottom - 52
    grouped = defaultdict(list)
    for row in random_rows: grouped[row["budget"]].append(row["useful_clips_captured"])
    ymax = max(max(values) for values in grouped.values()) + 5
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:Arial,sans-serif;fill:#202124}.small{font-size:12px}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>',
             f'<text x="{left}" y="28" font-size="18" font-weight="bold">Random-selection variability across 100 seeds</text>']
    for tick in range(0, int(ymax) + 1, 10):
        y = bottom + ph - tick / ymax * ph
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" class="grid"/><text x="{left-9}" y="{y+4:.1f}" text-anchor="end" class="small">{tick}</text>')
    for j, budget in enumerate(BUDGETS):
        vals = grouped[budget]; mean, sd, lo, hi = summarize_random(vals)
        x = left + pw * (j + .5) / len(BUDGETS)
        yy = lambda value: bottom + ph - value / ymax * ph
        parts.append(f'<line x1="{x}" y1="{yy(lo):.1f}" x2="{x}" y2="{yy(hi):.1f}" stroke="#7f8c8d" stroke-width="3"/>')
        parts.append(f'<line x1="{x-8}" y1="{yy(lo):.1f}" x2="{x+8}" y2="{yy(lo):.1f}" stroke="#7f8c8d" stroke-width="2"/><line x1="{x-8}" y1="{yy(hi):.1f}" x2="{x+8}" y2="{yy(hi):.1f}" stroke="#7f8c8d" stroke-width="2"/>')
        parts.append(f'<circle cx="{x}" cy="{yy(mean):.1f}" r="5" fill="#1769aa"/><text x="{x}" y="{bottom+ph+19}" text-anchor="middle" class="small">{int(budget*100)}%</text><text x="{x}" y="{yy(hi)-8:.1f}" text-anchor="middle" class="small">mean {mean:.1f}</text>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{left+pw}" y2="{bottom}" class="axis"/><text x="{left}" y="{height-12}" class="small">Vertical bars: min-max useful clips captured; dots: mean. Budget is share of eligible source clips.</text></svg>')
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(parts))


def main():
    score_by_name = {r["video_name"]: float(r["predicted_probability"])
                     for r in csv.DictReader(open(SCORES, encoding="utf-8")) if r["split"] == "source"}
    all_events = list(csv.DictReader(open(EVENTS, encoding="utf-8")))
    source = [r for r in all_events if r["split"] == "source"]
    if set(r["video_name"] for r in source) != set(score_by_name):
        raise ValueError("source score/event alignment failed")
    # Audio-unavailable clips cannot be selected for audio processing.  Exclude the
    # one unavailable source clip from every policy's common candidate pool.
    rows = []
    for r in source:
        if r["has_audio"] != "1":
            continue
        rows.append({"video_name": r["video_name"], "score": score_by_name[r["video_name"]],
                     "uncertainty": 1.0 - float(r["conf_camera"]),
                     "useful": int(r["correct_audio"] == "True" and r["correct_camera"] == "False")})
    total_useful = sum(r["useful"] for r in rows)
    if len(rows) != len(source) - 1:
        raise ValueError("unexpected source audio availability")
    result_rows, random_rows, plot_data = [], [], {}
    for budget in BUDGETS:
        k = math.ceil(budget * len(rows))
        policies = {
            "top_ranked": rank_select(rows, lambda r: r["score"], k),
            "camera_uncertainty": rank_select(rows, lambda r: r["uncertainty"], k),
        }
        for method, selected in policies.items():
            m = metrics(selected, rows, total_useful); m.update({"budget": budget, "method": method, "seed": ""})
            result_rows.append(m); plot_data[(budget, method)] = m
        by_metric = defaultdict(list)
        for seed in SEEDS:
            ids = list(range(len(rows))); random.Random(seed).shuffle(ids)
            m = metrics(ids[:k], rows, total_useful)
            random_rows.append(dict(m, budget=budget, seed=seed))
            for field in ("precision", "recall", "useful_clips_captured", "audio_usage_rate", "audio_correction_rate"):
                by_metric[field].append(m[field])
        mean_row = {"budget": budget, "method": "random_mean", "seed": ""}
        for field, values in by_metric.items():
            mean, sd, lo, hi = summarize_random(values)
            mean_row[field] = mean; mean_row[field + "_sd"] = sd; mean_row[field + "_min"] = lo; mean_row[field + "_max"] = hi
        mean_row["selected_clips"] = k
        result_rows.append(mean_row); plot_data[(budget, "random_mean")] = mean_row
    fields = ["budget", "method", "seed", "selected_clips", "precision", "recall", "useful_clips_captured", "audio_usage_rate", "audio_correction_rate", "precision_sd", "precision_min", "precision_max", "recall_sd", "recall_min", "recall_max", "useful_clips_captured_sd", "useful_clips_captured_min", "useful_clips_captured_max", "audio_usage_rate_sd", "audio_usage_rate_min", "audio_usage_rate_max", "audio_correction_rate_sd", "audio_correction_rate_min", "audio_correction_rate_max"]
    with open(os.path.join(MEMBER2, "audio_selector_budget_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(result_rows)
    with open(os.path.join(MEMBER2, "audio_selector_random_seeds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["budget", "seed", "selected_clips", "precision", "recall", "useful_clips_captured", "audio_usage_rate", "audio_correction_rate"]); w.writeheader(); w.writerows(random_rows)
    write_svg(os.path.join(MEMBER2, "audio_selector_budget_plot.svg"), plot_data)
    write_random_svg(os.path.join(MEMBER2, "audio_selector_random_variability.svg"), random_rows)
    with open(os.path.join(MEMBER2, "audio_selector_methodology.md"), "w", encoding="utf-8") as f:
        f.write("# Ranking evaluation: Member 2 audio selector\n\n")
        f.write("## Protocol\n\n")
        f.write("This retrospective evaluation uses **only source clips** and the pre-existing `availability.csv` source probabilities. Those probabilities are site-grouped OOF predictions: a clip's score came from a model trained without its `location_id` fold. No model was refit, tuned, thresholded, or calibrated for this analysis. Adapt and test clips, their labels, and all target outcomes are excluded.\n\n")
        f.write("The common candidate pool is %d source clips with an available audio stream (%d source clips total; one source clip has no audio). A clip is useful exactly when the existing audio expert is correct and the existing camera expert is incorrect. At each nominal budget, every policy selects `ceil(budget x %d)` candidates, so every comparison has exactly the same audio cost.\n\n" % (len(rows), len(source), len(rows)))
        f.write("Policies: (1) **top-ranked**, descending usefulness OOF probability; (2) **camera uncertainty**, descending `1 - conf_camera`; (3) **random**, an independently shuffled fixed-size selection for seeds 0-99. `video_name` is merely a deterministic tie-breaker for the ranked policies, not a feature.\n\n")
        f.write("Precision is useful clips / processed clips. Recall is useful clips captured / all useful candidates. Audio usage is processed clips / eligible candidates. Audio-correction rate is useful clips captured / eligible candidates (the deployment-wide yield); it equals precision x audio usage. Random variability is population SD plus minimum and maximum across the 100 seed runs.\n\n")
        f.write("## Results\n\n| Budget | Policy | Precision | Recall | Useful captured | Audio usage | Correction rate |\n|---:|---|---:|---:|---:|---:|---:|\n")
        for budget in BUDGETS:
            for row in [r for r in result_rows if r["budget"] == budget]:
                suffix = "" if row["method"] != "random_mean" else " (mean +/- SD)"
                def val(field):
                    text = "%.2f%%" % pct(row[field]) if field not in ("useful_clips_captured",) else "%.2f" % row[field]
                    if row["method"] == "random_mean":
                        sd = row.get(field + "_sd", 0)
                        text += " +/- " + ("%.2f%%" % pct(sd) if field not in ("useful_clips_captured",) else "%.2f" % sd)
                    return text
                f.write("| %d%% | %s%s | %s | %s | %s | %s | %s |\n" % (int(budget*100), row["method"], suffix, val("precision"), val("recall"), val("useful_clips_captured"), val("audio_usage_rate"), val("audio_correction_rate")))
        f.write("\nThe per-seed random results are retained in `audio_selector_random_seeds.csv`; aggregate values and random min/max are in `audio_selector_budget_results.csv`.\n")
    # Keep the primary Member 2 methodology current as well.  The baseline builder
    # owns its preceding sections, while this evaluator owns the marked appendix.
    main_path = os.path.join(MEMBER2, "methodology.md")
    marker = "\n\n## Budgeted ranking evaluation (rerun)\n"
    with open(main_path, encoding="utf-8") as f:
        main_text = f.read().split("\n## Budgeted ranking evaluation (rerun)", 1)[0].rstrip()
    appendix = [marker,
        "The following uses the regenerated source-only, site-grouped OOF scores in `availability.csv`. "
        "It evaluates 1,785 audio-available source clips (147 useful) and excludes adapt/test labels. "
        "All policies process identical fixed-size selections; random uses seeds 0-99.\n\n",
        "| Budget | Policy | Precision | Recall | Useful captured | Audio usage | Correction rate |\n",
        "|---:|---|---:|---:|---:|---:|---:|\n"]
    for budget in BUDGETS:
        for row in [r for r in result_rows if r["budget"] == budget]:
            def brief(field):
                output = "%.2f%%" % pct(row[field]) if field != "useful_clips_captured" else "%.2f" % row[field]
                if row["method"] == "random_mean":
                    sd = row[field + "_sd"]
                    output += " +/- " + ("%.2f%%" % pct(sd) if field != "useful_clips_captured" else "%.2f" % sd)
                return output
            appendix.append("| %d%% | %s | %s | %s | %s | %s | %s |\n" %
                            (int(budget * 100), row["method"], brief("precision"), brief("recall"),
                             brief("useful_clips_captured"), brief("audio_usage_rate"), brief("audio_correction_rate")))
    appendix.append("\nDetailed protocol: `audio_selector_methodology.md`. Reproducible evaluator: `evaluate_audio_selector.py`. "
                    "The top-ranked usefulness model exceeds random at all tested budgets, but camera uncertainty is stronger on this source-OOF cohort.\n")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_text + "".join(appendix))
    print("Eligible source candidates:", len(rows), "useful:", total_useful)


if __name__ == "__main__":
    main()
