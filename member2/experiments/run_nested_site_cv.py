"""Nested, site-grouped comparison of pre-audio usefulness selectors.

Run from repository root with .venv-member2\\Scripts\\python.exe
member2/experiments/run_nested_site_cv.py.  This script is intentionally
self-contained and never writes to Member 2's original baseline artifacts.
"""
import ast, csv, json, math, os, pickletools, random, struct, zipfile
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "safari", "outputs")
OUT = os.path.join(HERE, "outputs")
FOLDS = ["0", "1", "2", "3", "4"]
BUDGETS = (.10, .20, .30, .50)
SEEDS = range(100)
RNG = 20261115

def load_npy(handle):
    if handle.read(6) != b"\x93NUMPY": raise ValueError("not an NPY array")
    major, _ = handle.read(2); n = struct.unpack("<H" if major == 1 else "<I", handle.read(2 if major == 1 else 4))[0]
    h = ast.literal_eval(handle.read(n).decode("latin1").strip()); shape, desc = h["shape"], h["descr"]
    raw = handle.read(); count = math.prod(shape)
    if desc == "<f4": values = struct.unpack("<%df" % count, raw)
    elif desc == "<f8": values = struct.unpack("<%dd" % count, raw)
    elif desc == "|O": values = [x for op, x, _ in pickletools.genops(raw) if isinstance(x, str) and x.startswith("sa_fari_")]
    else: raise ValueError("unsupported dtype " + desc)
    if len(values) != count: raise ValueError("NPY decoding mismatch")
    return np.asarray(values).reshape(shape)

def npz(path, name):
    with zipfile.ZipFile(path) as z:
        with z.open(name + ".npy") as f: return load_npy(f)

def ap(y, p): return average_precision_score(y, p)
def score_metrics(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return {"roc_auc": roc_auc_score(y,p), "average_precision": ap(y,p), "brier": brier_score_loss(y,p), "log_loss": log_loss(y,p)}

def calibrator_fit(raw, y):
    # Platt scaling is fitted only to inner cross-fitted training predictions.
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(np.asarray(raw).reshape(-1,1), y)

def model(kind, params, seed):
    if kind.startswith("lr"):
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], solver="lbfgs", max_iter=2000, random_state=seed))
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=params["n_estimators"],
        max_depth=params["max_depth"], learning_rate=.05, min_child_weight=5, subsample=.8,
        colsample_bytree=.8, reg_lambda=10, reg_alpha=.2, random_state=seed, n_jobs=1, verbosity=0)

GRIDS = {
 "lr_camera": [{"C":v} for v in (.03,.1,.3,1)],
 "lr_contextual": [{"C":v} for v in (.03,.1,.3,1)],
 "xgb_contextual": [{"max_depth":d,"n_estimators":n} for d in (1,2) for n in (30,60)],
}

def inner_oof(kind, X, y, groups, params):
    """OOF predictions inside one outer-training partition (no outer labels)."""
    pred = np.zeros(len(y)); splitter = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    for n, (tr, va) in enumerate(splitter.split(X, y, groups)):
        m = model(kind, params, RNG+n); m.fit(X[tr], y[tr]); pred[va] = m.predict_proba(X[va])[:,1]
    return pred

def select_params(kind, X, y, groups):
    candidates=[]
    for params in GRIDS[kind]:
        p=inner_oof(kind,X,y,groups,params)
        candidates.append((ap(y,p), -params.get("C", 0), json.dumps(params,sort_keys=True),params))
    # AP is a ranking objective; stable lexical tie-break avoids validation peeking.
    return max(candidates)[3], candidates

def ranked(ids, score, k, names): return sorted(ids, key=lambda i:(-score[i], names[i]))[:k]
def outcome(sel, useful, n, denom):
    c=int(useful[sel].sum()); return dict(audio_calls=len(sel), useful_captured=c, precision=c/len(sel), recall=c/denom, correction_rate=c/n)

def main():
    os.makedirs(OUT, exist_ok=True)
    events=list(csv.DictReader(open(os.path.join(DATA,"clip_events.csv"),encoding="utf-8")))
    index=list(csv.DictReader(open(os.path.join(DATA,"feat_index.csv"),encoding="utf-8")))
    visual=load_npy(open(os.path.join(DATA,"visual_emb.npy"),"rb"))
    pn, cam=npz(os.path.join(DATA,"clip_probs.npz"),"video_name"),npz(os.path.join(DATA,"clip_probs.npz"),"camera")
    if [r["video_name"] for r in events] != [r["video_name"] for r in index] or len(events)!=len(visual): raise ValueError("visual/event alignment failed")
    if len(set(pn)) != len(pn) or set(pn) != {r["video_name"] for r in events}: raise ValueError("camera join failed")
    camera={str(n):np.asarray(p,dtype=float) for n,p in zip(pn,cam)}
    # `feat_index` was verified against events above; select embeddings by the
    # video-name key, never by the source subset's incidental row positions.
    visual_by_name = {r["video_name"]: visual[i] for i, r in enumerate(events)}
    source=[r for r in events if r["split"]=="source"]
    names=np.asarray([r["video_name"] for r in source]); folds=np.asarray([r["cv_fold"] for r in source]); groups=np.asarray([r["location_id"] for r in source])
    if sorted(set(folds)) != FOLDS: raise ValueError("unexpected outer folds")
    for f in FOLDS:
        if set(groups[folds==f]) & set(groups[folds!=f]): raise ValueError("site leakage in outer fold "+f)
    y=np.asarray([int(r["has_audio"]=="1" and r["correct_audio"]=="True" and r["correct_camera"]=="False") for r in source])
    cp=np.vstack([camera[n] for n in names]); ordered=np.sort(cp,axis=1)[:,::-1]
    entropy=-(cp*np.log(np.clip(cp,1e-12,1))).sum(1); camfeat=np.column_stack([cp,entropy,ordered[:,0]-ordered[:,1]])
    source_visual=np.vstack([visual_by_name[n] for n in names])
    pool=source_visual.reshape(len(source),32,16).mean(2)
    placement=np.asarray([[r["placement"]==v for v in ("arboreal","terrestrial","unknown")] for r in source],float)
    context=np.column_stack([camfeat, [r["is_ir"]=="True" for r in source], placement, pool])
    features={"lr_camera":camfeat,"lr_contextual":context,"xgb_contextual":context}
    raw={k:np.zeros(len(y)) for k in features}; calibrated={k:np.zeros(len(y)) for k in features}; chosen=[]
    # E1 has no fitted selector. Its scale conversion for global pooling is still
    # cross-fitted entirely within each outer training partition.
    raw["camera_uncertainty"]=1-cp.max(1); calibrated["camera_uncertainty"]=np.zeros(len(y))
    for fold_num,f in enumerate(FOLDS):
        tr=np.where(folds!=f)[0]; va=np.where(folds==f)[0]
        # E1 calibration learns from inner-grouped OOF uncertainty scores only.
        cal=calibrator_fit(raw["camera_uncertainty"][tr],y[tr]); calibrated["camera_uncertainty"][va]=cal.predict_proba(raw["camera_uncertainty"][va,None])[:,1]
        for kind,X in features.items():
            params, trials=select_params(kind,X[tr],y[tr],groups[tr]); inner=inner_oof(kind,X[tr],y[tr],groups[tr],params)
            cal=calibrator_fit(inner,y[tr]); final=model(kind,params,RNG+10+fold_num); final.fit(X[tr],y[tr])
            raw[kind][va]=final.predict_proba(X[va])[:,1]; calibrated[kind][va]=cal.predict_proba(raw[kind][va,None])[:,1]
            chosen.append({"outer_fold":f,"method":kind,"selected":json.dumps(params,sort_keys=True),"inner_trials":json.dumps([{ "params":t[3],"inner_ap":t[0]} for t in trials])})
    methods=["camera_uncertainty","lr_camera","lr_contextual","xgb_contextual"]
    eligible=np.asarray([r["has_audio"]=="1" for r in source]); ids=np.where(eligible)[0]; useful=y.copy(); useful[~eligible]=0; total=int(useful[ids].sum())
    score_rows=[]; metric_rows=[]; fold_rows=[]; budget_rows=[]
    for method in methods:
        for i in range(len(y)): score_rows.append(dict(video_name=names[i],outer_fold=folds[i],eligible_audio=int(eligible[i]),usefulness=int(useful[i]),method=method,raw_score=raw[method][i],calibrated_score=calibrated[method][i]))
        for label,sc in (("raw",raw[method]),("calibrated",calibrated[method])):
            m=score_metrics(useful[ids],sc[ids]); metric_rows.append(dict(method=method,score_variant=label,**m))
        for f in FOLDS:
            ii=np.where((folds==f)&eligible)[0]; m=score_metrics(useful[ii],calibrated[method][ii]); fold_rows.append(dict(method=method,outer_fold=f,n=len(ii),positives=int(useful[ii].sum()),**m))
        for strategy,sc in (("global_pooled_calibrated",calibrated[method]),("fold_stratified_raw",raw[method])):
            for b in BUDGETS:
                if strategy.startswith("global"):
                    sel=ranked(ids,sc,math.ceil(b*len(ids)),names)
                else:
                    sel=np.concatenate([ranked(np.where((folds==f)&eligible)[0],sc,math.ceil(b*sum((folds==f)&eligible)),names) for f in FOLDS])
                budget_rows.append(dict(method=method,strategy=strategy,budget=b,**outcome(sel,useful,len(ids),total)))
    # Common random selections for both strategies; deterministic and all eligible candidates.
    for strategy in ("global_pooled_calibrated","fold_stratified_raw"):
      for b in BUDGETS:
       for seed in SEEDS:
        rr=random.Random(seed)
        if strategy.startswith("global"):
            a=list(ids); rr.shuffle(a); sel=np.asarray(a[:math.ceil(b*len(ids))])
        else:
            chunks=[]
            for f in FOLDS:
                a=list(np.where((folds==f)&eligible)[0]); rr.shuffle(a); chunks+=a[:math.ceil(b*len(a))]
            sel=np.asarray(chunks)
        budget_rows.append(dict(method="random",strategy=strategy,budget=b,seed=seed,**outcome(sel,useful,len(ids),total)))
    def write(name, rows):
        with open(os.path.join(OUT,name),"w",newline="",encoding="utf-8") as h:
            w=csv.DictWriter(h,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    write("oof_scores.csv",score_rows); write("ranking_metrics.csv",metric_rows); write("fold_metrics.csv",fold_rows); write("budget_results.csv",budget_rows); write("tuning_choices.csv",chosen)
    # Site bootstrap of calibrated ranking metrics: resample location IDs, not clips.
    boot=[]; sites=np.unique(groups[eligible]); rng=np.random.default_rng(RNG)
    for method in methods:
      sc=calibrated[method]
      vals=defaultdict(list)
      for _ in range(1000):
        draw=rng.choice(sites,len(sites),replace=True); ii=np.concatenate([np.where((groups==s)&eligible)[0] for s in draw])
        for k,v in score_metrics(useful[ii],sc[ii]).items(): vals[k].append(v)
      for k,v in vals.items(): boot.append(dict(method=method,metric=k,ci_low=np.quantile(v,.025),ci_high=np.quantile(v,.975),bootstrap_replicates=1000))
    write("site_bootstrap_ci.csv",boot)
    with open(os.path.join(OUT,"report.md"),"w",encoding="utf-8") as h:
      h.write("# Member 2 nested site-grouped selector experiment\n\n")
      h.write("## Setup and leakage controls\n\nE1 ranks `1-max(camera posterior)` without model fitting. E2 uses camera posterior, entropy and top-two margin. E3 adds `is_ir`, one-hot placement and a fixed 32-D mean pool of the visual embedding. E4 is conservative XGBoost with the exact E3 inputs. All joins are by `video_name`; no audio-derived field, correctness field, label, site ID, target/adapt row, or row order is a model input. Source fitting includes only source rows. Evaluation uses the common 1,785 audio-available source clips (147 useful).\n\n")
      h.write("## Nested validation and calibration\n\nThe five supplied site folds are outer folds and were checked for zero train/validation location overlap. Within every outer training set, GroupKFold by location creates inner OOF predictions for conservative grid selection by AP and for Platt calibration. The final outer model is fitted on the full outer training set; its validation labels are never used for tuning or calibration. E1's scale conversion uses the same inner-only procedure. Global pooled ranking uses calibrated OOF probabilities to address fold scale; fold-stratified ranking uses raw scores independently inside each outer fold. XGBoost uses shallow trees, 30/60 rounds, depth 1/2, strong L2/min-child regularisation and no early stopping (avoiding an additional held-out tuning role with only 147 positives).\n\n")
      h.write("## OOF ranking and calibration\n\n| Method | Variant | ROC-AUC | AP | Brier | Log loss |\n|---|---|---:|---:|---:|---:|\n")
      for r in metric_rows: h.write("| %s | %s | %.4f | %.4f | %.4f | %.4f |\n"%(r['method'],r['score_variant'],r['roc_auc'],r['average_precision'],r['brier'],r['log_loss']))
      h.write("\nSite-resampled 95% confidence intervals and per-fold paired-comparison inputs are in `site_bootstrap_ci.csv` and `fold_metrics.csv`; they should be used instead of treating clips within a site as independent.\n\n## Budget results\n\n`budget_results.csv` holds every deterministic method result and all 100 random seeds. For each strategy/budget, random can be summarized directly across `seed=0..99`; precision, recall, useful captured and correction rate use the requested definitions. Fold-stratified selection can differ slightly in audio calls due to integer rounding per fold.\n\n## Interpretation and limitations\n\nUse the saved OOF tables for conclusions: complexity is not treated as a win by default. Results are limited by 147 positives, dependent expert-derived labels, and the finite set of sites. Calibration is source-OOF calibration only and is not claimed to transfer to target projects.\n")
    print("Wrote nested-CV experiment outputs to", OUT)

if __name__ == "__main__": main()
