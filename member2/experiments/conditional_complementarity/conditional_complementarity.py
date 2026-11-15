"""Conditional, retrospective camera/audio complementarity analysis.

Run from repository root:
  .venv-member2\\Scripts\\python.exe member2/experiments/conditional_complementarity/conditional_complementarity.py

All joins are on video_name.  Source weighted-fusion results are out-of-fold:
the 0.95 camera / 0.05 audio weight is fixed, so the supplied source folds only
define evaluation partitions (no labels are used to select a subgroup or weight).
"""
from __future__ import annotations
import csv, os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[3]
DATA, OUT = ROOT / "safari" / "outputs", Path(__file__).resolve().parent
RNG = np.random.default_rng(20261115)
EPS = 1e-12

def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f: return list(csv.DictReader(f))
def write_csv(name, rows):
    keys = sorted({k for r in rows for k in r})
    with open(OUT / name, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
def norm(p):
    p = np.clip(np.asarray(p, float), EPS, None)
    return p / p.sum(axis=1, keepdims=True)
def ece(y, p, bins=10):
    p, pred = norm(p), norm(p).argmax(1); conf = p.max(1); total = 0.
    for lo in np.linspace(0, 1, bins, endpoint=False):
        ix = (conf >= lo) & ((conf < lo + 1 / bins) if lo < .9 else (conf <= 1))
        if ix.any(): total += ix.mean() * abs((pred[ix] == y[ix]).mean() - conf[ix].mean())
    return total
def metrics(y, p):
    p = norm(p); pred = p.argmax(1); onehot = np.eye(p.shape[1])[y]
    return {"macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, pred),
            "log_loss": log_loss(y, p, labels=np.arange(p.shape[1])),
            "brier": np.mean((p-onehot)**2), "ece": ece(y,p),
            "mean_true_class_probability": p[np.arange(len(y)), y].mean()}
def quartiles(x):
    # qcut-like ranked bins keeps four populated, deterministic groups despite ties.
    order = np.argsort(x, kind="mergesort"); out = np.empty(len(x), int)
    out[order] = np.minimum(4, np.arange(len(x))*4//len(x)+1); return out
def subgroup_rows(pop, subgroup_kind, subgroup, y, cam, aud, sites):
    rows=[]; cp=cam.argmax(1); camera_metrics=metrics(y,cam)
    for method, p in (("camera",cam),("weighted_95_05",.95*cam+.05*aud),("equal_50_50",.5*cam+.5*aud)):
        p=norm(p); pred=p.argmax(1); changed=pred != cp; changed_correct=(pred[changed] == y[changed]).mean() if changed.any() else np.nan
        current=metrics(y,p)
        rows.append({"population":pop,"subgroup_kind":subgroup_kind,"subgroup":str(subgroup),
          "method":method,"n":len(y),"n_sites":len(set(sites)),**current,
          "macro_f1_difference_vs_camera":current["macro_f1"]-camera_metrics["macro_f1"],
          "accuracy_difference_vs_camera":current["accuracy"]-camera_metrics["accuracy"],
          "log_loss_difference_vs_camera":current["log_loss"]-camera_metrics["log_loss"],
          "brier_difference_vs_camera":current["brier"]-camera_metrics["brier"],
          "ece_difference_vs_camera":current["ece"]-camera_metrics["ece"],
          "mean_true_class_probability_difference_vs_camera":current["mean_true_class_probability"]-camera_metrics["mean_true_class_probability"],
          "mean_probability_original_camera_class":p[np.arange(len(y)),cp].mean(),
          "prediction_change_rate":changed.mean(),"changed_predictions_become_correct":changed_correct,
          "camera_correct_made_worse_rate":((cp==y)&(pred!=y)).mean(),
          "camera_incorrect_corrected_rate":((cp!=y)&(pred==y)).mean()})
    return rows
def bootstrap(pop, group_type, group, y, cam, fused, sites, reps=1000):
    # Site multiplicities are converted to clip weights, avoiding construction
    # of a large duplicated-clip array for every resample.
    unique, site_code=np.unique(sites, return_inverse=True); vals=[]
    cp, fp = cam.argmax(1), fused.argmax(1); nclass=cam.shape[1]
    cam_ll=-np.log(norm(cam)[np.arange(len(y)),y]); fused_ll=-np.log(norm(fused)[np.arange(len(y)),y])
    cam_tp=norm(cam)[np.arange(len(y)),y]; fused_tp=norm(fused)[np.arange(len(y)),y]
    def macro_f1(pred, weights):
        # Equivalent to multiclass macro-F1 over classes occurring in either
        # truth or predictions in the bootstrap sample.
        support=np.bincount(y,weights=weights,minlength=nclass); predicted=np.bincount(pred,weights=weights,minlength=nclass)
        correct = y == pred
        true_positive=np.bincount(y[correct], weights=weights[correct], minlength=nclass)
        denom=support+predicted; score=np.divide(2*true_positive,denom,out=np.zeros(nclass),where=denom>0)
        return score[denom>0].mean()
    for _ in range(reps):
        multiplicity=np.bincount(RNG.integers(0,len(unique),len(unique)),minlength=len(unique)); w=multiplicity[site_code]; den=w.sum()
        vals.append((macro_f1(fp,w)-macro_f1(cp,w), np.dot(w,fused_ll-cam_ll)/den,
                     np.dot(w,fused_tp-cam_tp)/den))
    vals=np.asarray(vals); rows=[]
    for j, metric in enumerate(("macro_f1_difference", "log_loss_difference", "mean_true_class_probability_difference")):
        rows.append({"population":pop,"group_type":group_type,"group":str(group),"comparison":"weighted_95_05_minus_camera",
          "metric":metric,"n":len(y),"n_sites":len(unique),"bootstrap_replicates":reps,
          "estimate":(metrics(y,fused)["macro_f1"]-metrics(y,cam)["macro_f1"] if j==0 else metrics(y,fused)["log_loss"]-metrics(y,cam)["log_loss"] if j==1 else metrics(y,fused)["mean_true_class_probability"]-metrics(y,cam)["mean_true_class_probability"]),
          "ci_low":np.quantile(vals[:,j],.025),"ci_high":np.quantile(vals[:,j],.975)})
    return rows
def safe(s): return str(s).strip().lower() in {"true","1","yes"}
def main():
    events={r["video_name"]:r for r in read_csv(DATA/"clip_events.csv")}
    with np.load(DATA/"clip_probs.npz",allow_pickle=True) as z:
        names=np.asarray(z["video_name"],str); classes=np.asarray(z["classes"],str); camera=z["camera"]; audio=z["audio"]
    if len(set(names)) != len(names) or set(names) != set(events) or camera.shape != audio.shape or camera.shape[1] != len(classes):
        raise ValueError("video_name or posterior dimensions failed validation")
    cidx={c:i for i,c in enumerate(classes)}; data=[]
    for i,n in enumerate(names):
        r=events[n]
        if r["species"] in cidx: data.append((r,cidx[r["species"]],camera[i],audio[i]))
    output={"confidence_analysis.csv":[],"margin_analysis.csv":[],"environment_analysis.csv":[],"species_analysis.csv":[],"domain_shift_analysis.csv":[],"bootstrap_intervals.csv":[]}
    populations={p:[x for x in data if x[0]["split"]==p] for p in ("source","adapt","test")}
    if not populations["source"]: raise ValueError("No label-supported source rows")
    def arrays(d): return (np.array([x[1] for x in d]),np.stack([x[2] for x in d]),np.stack([x[3] for x in d]),np.array([x[0]["location_id"] for x in d]),[x[0] for x in d])
    # A/B: source OOF posterior inputs; fixed weights need no fitting. Quartile boundaries are descriptive.
    y,cam,aud,site,meta=arrays(populations["source"]); conf=cam.max(1); mar=np.sort(cam,axis=1)[:,-1]-np.sort(cam,axis=1)[:,-2]
    for label, values, dest in (("camera_confidence",conf,"confidence_analysis.csv"),("camera_margin",mar,"margin_analysis.csv")):
        q=quartiles(values)
        for k in range(1,5):
            mask=q==k
            output[dest] += subgroup_rows("source_oof",label,f"Q{k}",y[mask],cam[mask],aud[mask],site[mask])
            output["bootstrap_intervals.csv"] += bootstrap("source_oof",label,f"Q{k}",y[mask],cam[mask],norm(.95*cam[mask]+.05*aud[mask]),site[mask])
    # C: source environmental observational subgroups. Unknown placement remains explicit.
    env=[]
    for ir in ("IR","non_IR"):
        mask=np.array([safe(r["is_ir"]) if ir=="IR" else not safe(r["is_ir"]) for r in meta]); env.append(("is_ir",ir,mask))
    for place in ("arboreal","terrestrial","unknown"):
        mask=np.array([(r.get("placement") or "unknown").strip().lower()==place for r in meta]); env.append(("placement",place,mask))
    for typ,label,mask in env:
        if not mask.any(): continue
        output["environment_analysis.csv"] += subgroup_rows("source_oof",typ,label,y[mask],cam[mask],aud[mask],site[mask])
        output["bootstrap_intervals.csv"] += bootstrap("source_oof",typ,label,y[mask],cam[mask],norm(.95*cam[mask]+.05*aud[mask]),site[mask])
    # D: post-hoc species results; n<20 is flagged, no selector feature is constructed.
    fused=norm(.95*cam+.05*aud)
    for k,name in enumerate(classes):
        ix=y==k
        if not ix.any(): continue
        output["species_analysis.csv"].append({"population":"source_oof","species":name,"n":ix.sum(),"n_sites":len(set(site[ix])),
          "low_sample_flag": "insufficient_n_lt_20" if ix.sum()<20 else "", "camera_accuracy":(cam[ix].argmax(1)==y[ix]).mean(),
          "audio_accuracy":(aud[ix].argmax(1)==y[ix]).mean(),"weighted_fusion_accuracy":(fused[ix].argmax(1)==y[ix]).mean(),
          "camera_mean_true_class_probability":cam[ix,np.full(ix.sum(),k)].mean(),"fusion_mean_true_class_probability":fused[ix,np.full(ix.sum(),k)].mean(),
          "fusion_minus_camera_log_loss":metrics(y[ix],fused[ix])["log_loss"]-metrics(y[ix],cam[ix])["log_loss"]})
    # E: populations separately; target is evaluated with the same source-derived fixed 0.95 weight, never target-tuned.
    for pop,d in populations.items():
        yy,cc,aa,ss,_=arrays(d); ff=norm(.95*cc+.05*aa); mc,mf=metrics(yy,cc),metrics(yy,ff)
        output["domain_shift_analysis.csv"].append({"population":"source_oof" if pop=="source" else f"target_{pop}","n":len(yy),"n_sites":len(set(ss)),
          **{f"camera_{k}":v for k,v in mc.items()},**{f"weighted_95_05_{k}":v for k,v in mf.items()},
          "macro_f1_difference":mf["macro_f1"]-mc["macro_f1"],"accuracy_difference":mf["accuracy"]-mc["accuracy"],"log_loss_difference":mf["log_loss"]-mc["log_loss"],"mean_true_class_probability_difference":mf["mean_true_class_probability"]-mc["mean_true_class_probability"]})
        output["bootstrap_intervals.csv"] += bootstrap("source_oof" if pop=="source" else f"target_{pop}","population",pop,yy,cc,ff,ss)
    for name,rows in output.items(): write_csv(name,rows)
    def get(path, condition): return [r for r in output[path] if condition(r)]
    domain=output["domain_shift_analysis.csv"]
    src=domain[0]; q1=get("confidence_analysis.csv",lambda r:r["subgroup"]=="Q1" and r["method"]=="weighted_95_05")[0]; q4=get("confidence_analysis.csv",lambda r:r["subgroup"]=="Q4" and r["method"]=="weighted_95_05")[0]
    m1=get("margin_analysis.csv",lambda r:r["subgroup"]=="Q1" and r["method"]=="weighted_95_05")[0]
    species=sorted([r for r in output["species_analysis.csv"] if not r["low_sample_flag"]],key=lambda r:float(r["fusion_minus_camera_log_loss"]))[:5]
    report=f"""# Conditional multimodal complementarity

## Research question and data

Does the **existing audio classifier posterior** provide information beyond the existing camera posterior under particular conditions? This is a retrospective posterior-fusion analysis, not an acquisition policy. `clip_probs.npz` ({len(names)} unique `video_name` keys, {len(classes)} classes) was joined to `clip_events.csv` by `video_name`; no row order was used. Label-supported source has {len(y)} clips from {len(set(site))} sites. Source results use the supplied site-held-out predictions already produced by the current pipeline; fixed 95/5 and 50/50 arithmetic fusion was then applied without subgroup-specific tuning. Target adapt and test use the same source-derived 95/5 weight and are reported separately. Site/project IDs, species, annotations, and any audio-derived quantity are not model inputs.

## A. Camera-confidence quartiles

`confidence_analysis.csv` gives all requested metrics for camera, 95/5, and equal fusion. In the lowest-confidence quartile, 95/5 changes macro-F1 by {float(q1['macro_f1_difference_vs_camera']):+.3f} and log loss by {float(q1['log_loss_difference_vs_camera']):+.3f}; in the highest-confidence quartile it had macro-F1 {float(q4['macro_f1']):.3f}, accuracy {float(q4['accuracy']):.3f}, and a prediction-change rate of {float(q4['prediction_change_rate']):.3f}. Thus any apparent low-confidence top-1 gain should be treated cautiously alongside the multiple subgroup comparisons and its interval-free, descriptive nature. Quartiles are ranked into equal-sized groups to handle tied posterior values; no weights were selected within a group.

## B. Camera top-two-margin quartiles

`margin_analysis.csv` repeats the same complete comparison, including true-class probability, probability of the original camera class, changed-prediction correctness, correction, and harm rates. In the lowest-margin (most visually ambiguous) quartile, 95/5 changes macro-F1 by {float(m1['macro_f1_difference_vs_camera']):+.3f}, while log loss changes by {float(m1['log_loss_difference_vs_camera']):+.3f}. A higher maximum posterior is not interpreted as better confidence unless the true-class and outcome measures agree.

## C. Environmental subgroups

`environment_analysis.csv` reports IR/non-IR and arboreal/terrestrial/unknown-placement comparisons, with paired camera-to-fusion deltas. IR is treated only as an IR/low-light proxy, not as night. These observational subgroup differences are not causal claims.

## D. Species (post-hoc)

`species_analysis.csv` is retrospective only and flags fewer than 20 clips as insufficient. Among species with at least 20 source clips, the five largest fusion log-loss decreases are: {', '.join(f"{r['species']} ({float(r['fusion_minus_camera_log_loss']):+.3f})" for r in species)}. This ranking is exploratory and species identity must not become an acquisition-selector input.

## E. Population/domain comparison

| Population | n | sites | Camera macro-F1 | 95/5 macro-F1 | Difference | Camera log loss | 95/5 log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {r['population']} | {r['n']} | {r['n_sites']} | {float(r['camera_macro_f1']):.3f} | {float(r['weighted_95_05_macro_f1']):.3f} | {float(r['macro_f1_difference']):+.3f} | {float(r['camera_log_loss']):.3f} | {float(r['weighted_95_05_log_loss']):.3f} |" for r in domain) + f"""

The source headline is camera macro-F1 {float(src['camera_macro_f1']):.3f} versus 95/5 fusion {float(src['weighted_95_05_macro_f1']):.3f}; log loss changes by {float(src['log_loss_difference']):+.3f}. Target results are fixed-choice evaluations, not target-selected improvements. They do not justify a claim of improved domain robustness unless their paired differences and intervals are convincingly favorable.

## Statistical uncertainty and interpretation

`bootstrap_intervals.csv` contains 1,000 paired site-cluster bootstrap resamples for macro-F1, log loss, and mean true-class-probability differences, for each population, confidence/margin quartile, and environmental subgroup. Each resampled site carries all of its clips. These intervals condition on the existing modality predictions: they do **not** capture uncertainty from retraining either model. The many quartile, environment, and species comparisons are exploratory and should not be interpreted as confirmatory without multiple-comparison control and independent validation.

This can establish only whether this existing audio classifier's predictions add information beyond this existing camera classifier. It cannot show that raw audio lacks information, that a stronger audio model or representation-level fusion cannot help, or that audio can recover camera-untriggered events. Retrospective fusion also differs from an acquisition policy: both predictions are available here, whereas a policy must decide whether to process audio before its prediction exists. The next policy experiment should therefore use only permitted pre-audio signals and prospectively validate any pre-specified conditional benefit; absent robust conditional intervals, prioritizing camera adaptation is better supported.
"""
    (OUT/"conditional_complementarity_report.md").write_text(report,encoding="utf-8")
    (OUT/"README.md").write_text("# Reproduction\n\nFrom the repository root run:\n\n```powershell\n.venv-member2\\Scripts\\python.exe member2\\experiments\\conditional_complementarity\\conditional_complementarity.py\n```\n\nInputs are read-only: `safari/outputs/clip_probs.npz` and `clip_events.csv`. The script validates posterior/event keys and writes all deliverables in this directory. It requires NumPy and scikit-learn from `.venv-member2`.\n",encoding="utf-8")
    print(f"Wrote conditional complementarity deliverables to {OUT}")
if __name__ == "__main__": main()
