"""Reproducible Member 2 pre-audio usefulness baseline.

Uses only source clips to fit a site-grouped, OOF-evaluated logistic regression.
No third-party Python packages are required.
"""
import ast
import csv
import math
import os
import pickletools
import struct
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "safari", "outputs")
EPS = 1e-8


def load_npy(handle):
    if handle.read(6) != b"\x93NUMPY":
        raise ValueError("not an NPY array")
    major, _ = handle.read(2)
    n = struct.unpack("<H" if major == 1 else "<I", handle.read(2 if major == 1 else 4))[0]
    header = ast.literal_eval(handle.read(n).decode("latin1").strip())
    shape, desc = header["shape"], header["descr"]
    if header["fortran_order"]:
        raise ValueError("Fortran-order arrays are unsupported")
    count = math.prod(shape)
    raw = handle.read()
    if desc == "<f4": values = list(struct.unpack("<%df" % count, raw))
    elif desc == "<f8": values = list(struct.unpack("<%dd" % count, raw))
    elif desc.startswith("<U"):
        width = int(desc[2:]); values = [raw[i*width*4:(i+1)*width*4].decode("utf-32le").rstrip("\0") for i in range(count)]
    elif desc == "|O":
        # This repository's video-name array is an object NPY containing a pickle.
        # Extract only the serialized sa_fari strings; no untrusted pickle is executed.
        values = [arg for op, arg, _ in pickletools.genops(raw) if isinstance(arg, str) and arg.startswith("sa_fari_")]
        if len(values) != count: raise ValueError("could not safely recover object string array")
    else: raise ValueError("unsupported dtype " + desc)
    if len(shape) == 1: return values
    width = shape[-1]
    return [values[i*width:(i+1)*width] for i in range(shape[0])]


def npy(path):
    with open(path, "rb") as f: return load_npy(f)


def npz(path, name):
    with zipfile.ZipFile(path) as z:
        with z.open(name + ".npy") as f: return load_npy(f)


def sigmoid(z):
    z = max(-35., min(35., z))
    return 1. / (1. + math.exp(-z))


def standardize_fit(x, train):
    means = [sum(x[i][j] for i in train)/len(train) for j in range(len(x[0]))]
    scales = [(sum((x[i][j]-means[j])**2 for i in train)/len(train))**.5 for j in range(len(means))]
    return means, [s if s > EPS else 1. for s in scales]


def transform(x, means, scales):
    return [[(v-m)/s for v, m, s in zip(row, means, scales)] for row in x]


def fit_logistic(x, y, train, l2=0.30, epochs=120, lr=0.035):
    # Unweighted log loss: its output is a probability on the observed source prevalence.
    d = len(x[0]); w = [0.] * d
    positives = sum(y[i] for i in train)
    b = math.log(max(positives, EPS) / max(len(train) - positives, EPS))
    mw = [0.] * d; vw = [0.] * d; mb = vb = 0.
    for t in range(1, epochs + 1):
        gw = [l2 * v for v in w]; gb = 0.
        for i in train:
            p = sigmoid(b + sum(a*c for a, c in zip(w, x[i])))
            err = p - y[i]
            gb += err
            for j, v in enumerate(x[i]): gw[j] += err * v
        inv = 1. / len(train)
        for j in range(d):
            gw[j] *= inv; mw[j] = .9*mw[j] + .1*gw[j]; vw[j] = .999*vw[j] + .001*gw[j]*gw[j]
            w[j] -= lr * (mw[j]/(1-.9**t)) / (math.sqrt(vw[j]/(1-.999**t)) + EPS)
        gb *= inv; mb = .9*mb + .1*gb; vb = .999*vb + .001*gb*gb
        b -= lr * (mb/(1-.9**t)) / (math.sqrt(vb/(1-.999**t)) + EPS)
    return w, b


def auc(y, p):
    """Mann-Whitney ROC-AUC, assigning half credit to every tied pos/neg pair."""
    pairs = sorted(zip(p, y), key=lambda pair: pair[0]); n1 = sum(y); n0 = len(y)-n1
    if not n1 or not n0: return float("nan")
    wins = 0.; negatives_before = 0; i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        positives = sum(label for _, label in pairs[i:j]); negatives = (j-i) - positives
        wins += positives * negatives_before + .5 * positives * negatives
        negatives_before += negatives; i = j
    return wins / (n1*n0)


def average_precision(y, p):
    positives = sum(y)
    if not positives: return float("nan")
    ordered = sorted(zip(p, y), key=lambda pair: pair[0], reverse=True); hits = 0; score = 0.; i = 0
    # Each score threshold is evaluated as a group, avoiding label-order artefacts for tied predictions.
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]: j += 1
        group_hits = sum(label for _, label in ordered[i:j]); hits += group_hits
        score += group_hits * hits / j
        i = j
    return score / positives


def brier(y, p): return sum((label-prob)**2 for label, prob in zip(y, p)) / len(y)


def log_loss(y, p): return -sum(label*math.log(max(prob, EPS)) + (1-label)*math.log(max(1-prob, EPS)) for label, prob in zip(y, p)) / len(y)


def ece(y, p, bins=10):
    total = len(y); result = 0.
    for b in range(bins):
        lower, upper = b/bins, (b+1)/bins
        idx = [i for i, prob in enumerate(p) if lower <= prob < upper or (b == bins-1 and prob == upper)]
        if idx: result += len(idx)/total * abs(sum(p[i] for i in idx)/len(idx) - sum(y[i] for i in idx)/len(idx))
    return result


def main():
    events = list(csv.DictReader(open(os.path.join(DATA, "clip_events.csv"), newline="", encoding="utf-8")))
    index = list(csv.DictReader(open(os.path.join(DATA, "feat_index.csv"), newline="", encoding="utf-8")))
    visual = npy(os.path.join(DATA, "visual_emb.npy"))
    probs_file = os.path.join(DATA, "clip_probs.npz")
    prob_names, camera = npz(probs_file, "video_name"), npz(probs_file, "camera")
    event_by_name = {r["video_name"]: r for r in events}
    if len(event_by_name) != len(events) or len(index) != len(visual): raise ValueError("duplicate event names or visual-index mismatch")
    if [r["video_name"] for r in index] != [r["video_name"] for r in events]: raise ValueError("visual embeddings are not aligned to clip_events")
    if set(prob_names) != set(event_by_name) or len(prob_names) != len(camera): raise ValueError("camera probabilities do not align by video_name")
    prob_by_name = dict(zip(prob_names, camera))
    placements = ["arboreal", "terrestrial", "unknown"]
    x, y, names, rows = [], [], [], []
    for i, r in enumerate(events):
        cp = prob_by_name[r["video_name"]]
        ordered = sorted(cp, reverse=True)
        entropy = -sum(q*math.log(max(q, EPS)) for q in cp)
        # Permitted pre-audio features only: visual embedding, camera posterior/uncertainty, IR, placement.
        visual_pooled = [sum(visual[i][j:j+16]) / 16. for j in range(0, 512, 16)]
        feat = visual_pooled + cp + [entropy, ordered[0]-ordered[1], float(r["is_ir"] == "True")]
        feat += [float(r["placement"] == val) for val in placements]
        x.append(feat); names.append(r["video_name"]); rows.append(r)
        y.append(int(r["has_audio"] == "1" and r["correct_audio"] == "True" and r["correct_camera"] == "False"))
    source = [i for i, r in enumerate(rows) if r["split"] == "source"]
    folds = sorted(set(rows[i]["cv_fold"] for i in source))
    if folds != ["0", "1", "2", "3", "4"]: raise ValueError("expected five source OOF folds")
    for fold in folds:
        held = [i for i in source if rows[i]["cv_fold"] == fold]; train = [i for i in source if rows[i]["cv_fold"] != fold]
        train_sites = {rows[i]["location_id"] for i in train}; held_sites = {rows[i]["location_id"] for i in held}
        if train_sites & held_sites: raise ValueError("site leakage in fold " + fold)
    oof = [None] * len(rows)
    for fold in folds:
        held = [i for i in source if rows[i]["cv_fold"] == fold]; train = [i for i in source if rows[i]["cv_fold"] != fold]
        means, scales = standardize_fit(x, train); xs = transform(x, means, scales); w, b = fit_logistic(xs, y, train)
        for i in held: oof[i] = sigmoid(b + sum(a*c for a, c in zip(w, xs[i])))
    means, scales = standardize_fit(x, source); xs = transform(x, means, scales); w, b = fit_logistic(xs, y, source)
    pred = [oof[i] if i in source else sigmoid(b + sum(a*c for a, c in zip(w, xs[i]))) for i in range(len(rows))]
    # No audio is available to invoke: force no-use probability, while retaining a truthful zero outcome.
    for i, r in enumerate(rows):
        if r["has_audio"] != "1": pred[i] = 0.; y[i] = 0
    source_y, source_p = [y[i] for i in source], [pred[i] for i in source]
    prevalence = sum(source_y) / len(source_y)
    prevalence_p = [prevalence] * len(source_y)
    # Regression tests for tied-score behavior (including the all-tied prevalence baseline).
    if abs(auc([1, 0], [.5, .5]) - .5) > EPS or abs(average_precision(source_y, prevalence_p) - prevalence) > EPS:
        raise AssertionError("tied-score metric handling failed")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "availability.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "usefulness", "predicted_probability", "split"])
        writer.writeheader()
        for i, r in enumerate(rows): writer.writerow({"video_name": names[i], "usefulness": y[i], "predicted_probability": "%.8f" % pred[i], "split": r["split"]})
    with open(os.path.join(OUT, "methodology.md"), "w", encoding="utf-8") as f:
        f.write("# Member 2: Binary audio-usefulness baseline\n\n")
        f.write("`usefulness = 1` exactly when audio is correct and camera is incorrect. The source target uses the existing site-OOF expert outcomes; no target clips contribute to fitting or OOF model selection.\n\n")
        f.write("## Model and validation\n\nAn unweighted L2 logistic regression is fitted on source clips only. Features are a fixed 32-D mean-pooling projection of the 512-D visual embeddings, the full camera class posterior, camera entropy and top-two margin, `is_ir`, and one-hot placement. It uses the five supplied site-grouped OOF folds; every fold was checked to have zero location-ID overlap between train and validation. Source rows in `availability.csv` contain usefulness-model OOF probabilities. Adapt/test rows use a final source-only fit.\n\nThe previous class-balanced objective shifted its probability scale toward 50% (mean OOF prediction 38.86% versus 8.23% prevalence). It was removed: the emitted value is an unweighted logistic probability initialized at the training-fold prevalence, not a class-weighted ranking score. Its calibration is assessed below using source OOF labels only; no target labels are used for calibration.\n\n")
        f.write("## Source site-grouped OOF metrics\n\n")
        f.write("- Clips: %d; positives: %d (%.2f%%)\n" % (len(source), sum(source_y), 100*sum(source_y)/len(source)))
        f.write("- Logistic OOF: ROC-AUC %.4f; average precision %.4f; Brier %.4f; log loss %.4f; 10-bin ECE %.4f; mean prediction %.4f\n" % (auc(source_y, source_p), average_precision(source_y, source_p), brier(source_y, source_p), log_loss(source_y, source_p), ece(source_y, source_p), sum(source_p)/len(source_p)))
        f.write("- Prevalence-only OOF (%0.4f for every clip): ROC-AUC %.4f; average precision %.4f; Brier %.4f; log loss %.4f; 10-bin ECE %.4f\n" % (prevalence, auc(source_y, prevalence_p), average_precision(source_y, prevalence_p), brier(source_y, prevalence_p), log_loss(source_y, prevalence_p), ece(source_y, prevalence_p)))
        f.write("- ROC-AUC uses the Mann-Whitney formulation: a tied positive/negative pair receives 0.5 credit. Average precision is also evaluated by tied score threshold, so the all-tied prevalence baseline has AP equal to prevalence.\n")
        f.write("- Missing-audio clips: %d total (%d source); their label and probability are set to 0.\n" % (sum(r["has_audio"] != "1" for r in rows), sum(rows[i]["has_audio"] != "1" for i in source)))
        f.write("\n## Leakage and alignment checks\n\nPassed: unique video names; `feat_index`/visual embedding row alignment; camera probability names aligned to events; five expected source folds; and no train/validation site overlap. The implementation never reads audio embeddings, AudioSet probabilities, audio confidence/prediction as features, species/other labels as features, annotation/oracle fields, or `location_id` as a feature.\n\n")
        f.write("## Limitations\n\nThe target is derived from two imperfect expert predictions and inherits their errors. Visual embeddings may encode environment correlations despite site-grouped validation. Calibration is evaluated only on source OOF predictions (and is not assumed to transfer to a new project); a probability alone does not specify a deployment audio-cost threshold. Target usefulness is retained in the CSV only for retrospective evaluation, never fitting.\n")
    print("Wrote", os.path.join(OUT, "availability.csv"))


if __name__ == "__main__": main()
