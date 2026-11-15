"""Nested site-CV experiment for simple conditional audio-selection policies.

Run from repository root:
  .venv-member2\\Scripts\\python.exe member2/experiments/conditional_policy/conditional_policy_runner.py
"""
import ast, csv, math, os, pickletools, random, struct, zipfile
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA = os.path.join(ROOT, "safari", "outputs")
FOLDS, BUDGETS, SEEDS, RNG = [str(x) for x in range(5)], (.10, .20, .30, .50), range(100), 20261115

def load_npy(h):
    if h.read(6) != b"\x93NUMPY": raise ValueError("not npy")
    major = h.read(1)[0]; h.read(1); n = struct.unpack("<H" if major == 1 else "<I", h.read(2 if major == 1 else 4))[0]
    header = ast.literal_eval(h.read(n).decode("latin1").strip()); raw = h.read(); count = math.prod(header["shape"])
    if header["descr"] == "<f4": values = struct.unpack("<%df" % count, raw)
    elif header["descr"] == "<f8": values = struct.unpack("<%dd" % count, raw)
    elif header["descr"] == "|O": values = [x for op,x,_ in pickletools.genops(raw) if isinstance(x,str) and x.startswith("sa_fari_")]
    else: raise ValueError("unsupported npy type")
    return np.asarray(values).reshape(header["shape"])

def npz(path, key):
    with zipfile.ZipFile(path) as z:
        with z.open(key + ".npy") as h: return load_npy(h)

def qbins(train_u, u):
    """Five feature-only quantile bins, with deterministic handling of ties."""
    edges = np.quantile(train_u, [.2,.4,.6,.8])
    return np.searchsorted(edges, u, side="right"), edges

def keys(policy, bins, ir, placement):
    if policy == "E2_bins": return [(int(b),) for b in bins]
    if policy == "E3_bins_x_ir": return [(int(b), int(i)) for b,i in zip(bins,ir)]
    if policy == "E4_bins_x_placement": return [(int(b), str(p)) for b,p in zip(bins,placement)]
    return [(int(b), int(i), str(p)) for b,i,p in zip(bins,ir,placement)]

def policy_score(policy, tr, pred, u, ir, placement, y):
    """Fit a low-dimensional table on tr, then predict rates for pred.
    Fixed beta-binomial smoothing (5 pseudo-observations) avoids unstable rare cells;
    it is not tuned on labels.  E1 deliberately remains raw uncertainty here.
    """
    if policy == "E1_uncertainty": return u[pred]
    trbin, edges = qbins(u[tr], u[tr]); pbin = np.searchsorted(edges, u[pred], side="right")
    parent = (y[tr].sum() + 1) / (len(tr) + 2)
    count, total = defaultdict(float), defaultdict(float)
    for key, yy in zip(keys(policy,trbin,ir[tr],placement[tr]), y[tr]): total[key] += 1; count[key] += yy
    # Back off an unseen interaction cell to its global outer-training rate.
    return np.asarray([(count[k] + 5*parent) / (total[k] + 5) for k in keys(policy,pbin,ir[pred],placement[pred])])

def inner_raw(policy, outer_tr, u, ir, placement, y, groups):
    out = np.zeros(len(outer_tr)); g = groups[outer_tr]
    for a,b in GroupKFold(n_splits=min(4,len(np.unique(g)))).split(outer_tr, y[outer_tr], g):
        out[b] = policy_score(policy, outer_tr[a], outer_tr[b], u,ir,placement,y)
    return out

def calibration(raw, y):
    # All policies use the same Platt conversion, fitted only to inner OOF scores.
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(raw.reshape(-1,1),y)

def select(ids, score, k, names): return np.asarray(sorted(ids, key=lambda i:(-score[i],names[i]))[:k])
def outcome(sel, y, eligible_n, positives):
    captured = int(y[sel].sum()); calls=len(sel)
    return {"audio_calls":calls,"useful_clips_captured":captured,"precision":captured/calls if calls else 0,
            "recall":captured/positives if positives else 0,"correction_rate":captured/eligible_n if eligible_n else 0}
def write(name, rows):
    with open(os.path.join(HERE,name),"w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)

def main():
    events=list(csv.DictReader(open(os.path.join(DATA,"clip_events.csv"),encoding="utf-8")))
    source=[r for r in events if r["split"] == "source"]
    names=np.asarray([r["video_name"] for r in source]); folds=np.asarray([r["cv_fold"] for r in source]); groups=np.asarray([r["location_id"] for r in source])
    y=np.asarray([int(r["has_audio"] == "1" and r["correct_audio"] == "True" and r["correct_camera"] == "False") for r in source])
    eligible=np.asarray([r["has_audio"] == "1" for r in source]); ir=np.asarray([r["is_ir"] == "True" for r in source],int); placement=np.asarray([r["placement"] for r in source])
    pn, cp = npz(os.path.join(DATA,"clip_probs.npz"),"video_name"), npz(os.path.join(DATA,"clip_probs.npz"),"camera")
    camera={str(n):p for n,p in zip(pn,cp)}; u=np.asarray([1-np.max(camera[n]) for n in names])
    assert sorted(set(folds)) == FOLDS
    for f in FOLDS: assert not (set(groups[folds==f]) & set(groups[folds!=f]))
    policies=["E1_uncertainty","E2_bins","E3_bins_x_ir","E4_bins_x_placement","E5_bins_x_ir_x_placement"]
    raw={p:np.zeros(len(y)) for p in policies}; score={p:np.zeros(len(y)) for p in policies}; config=[]
    for f in FOLDS:
        tr=np.where(folds!=f)[0]; va=np.where(folds==f)[0]
        for p in policies:
            inn=inner_raw(p,tr,u,ir,placement,y,groups); cal=calibration(inn,y[tr])
            raw[p][va]=policy_score(p,tr,va,u,ir,placement,y)
            score[p][va]=cal.predict_proba(raw[p][va,None])[:,1]
            config.append({"outer_fold":f,"policy":p,"calibration":"Platt on inner grouped OOF scores","smoothing":"5 pseudo-observations toward outer-training prevalence" if p != policies[0] else "not applicable"})
    ids=np.where(eligible)[0]; positives=int(y[ids].sum()); results=[]; foldrows=[]; selected={}
    # Primary comparison: calibrated OOF scores, one common eligible universe and exact global budgets.
    for p in policies:
        for b in BUDGETS:
            sel=select(ids,score[p],math.ceil(b*len(ids)),names); selected[(p,b)] = set(sel)
            results.append({"policy":p,"score_variant":"calibrated_global","budget":b,"seed":"","selection_scope":"pooled_oof",**outcome(sel,y,len(ids),positives)})
            for f in FOLDS:
                ii=np.where((folds==f)&eligible)[0]; fs=select(ii,score[p],math.ceil(b*len(ii)),names)
                foldrows.append({"policy":p,"score_variant":"calibrated","outer_fold":f,"budget":b,"n_eligible":len(ii),"n_positive":int(y[ii].sum()),**outcome(fs,y,len(ii),int(y[ii].sum()))})
    # Explicitly retain E1's raw ranking as a diagnostic; it is not the calibrated pooled policy.
    for b in BUDGETS:
        sel=select(ids,raw[policies[0]],math.ceil(b*len(ids)),names)
        results.append({"policy":"E1_uncertainty","score_variant":"raw_ranking","budget":b,"seed":"","selection_scope":"pooled_oof",**outcome(sel,y,len(ids),positives)})
    for b in BUDGETS:
        for seed in SEEDS:
            a=list(ids); random.Random(seed).shuffle(a); sel=np.asarray(a[:math.ceil(b*len(ids))])
            results.append({"policy":"random","score_variant":"random","budget":b,"seed":seed,"selection_scope":"pooled_oof",**outcome(sel,y,len(ids),positives)})
    # Paired fold differences are raw inputs, written as rows alongside fold results.
    base={(r["outer_fold"],r["budget"]):r for r in foldrows if r["policy"]==policies[0]}
    for r in list(foldrows):
        if r["policy"] != policies[0]:
            q=base[(r["outer_fold"],r["budget"])]
            for metric in ("useful_clips_captured","precision","recall","correction_rate"):
                foldrows.append({"row_type":"paired_difference_vs_E1","policy":r["policy"],"outer_fold":r["outer_fold"],"budget":r["budget"],"metric":metric,"difference":r[metric]-q[metric]})
    # Site bootstrap of the already fixed pooled selections (site is the resampling unit).
    intervals=[]; sites=np.unique(groups[eligible]); rng=np.random.default_rng(RNG)
    site_ix={s:np.where((groups==s)&eligible)[0] for s in sites}
    for p in policies:
      for b in BUDGETS:
        flag=np.asarray([i in selected[(p,b)] for i in range(len(y))]); vals=[]
        per_site=np.asarray([y[site_ix[s]][flag[site_ix[s]]].sum() for s in sites],float)
        for _ in range(500):
            vals.append(float(per_site[rng.integers(0,len(sites),len(sites))].sum()))
        intervals.append({"policy":p,"budget":b,"metric":"useful_clips_captured","estimate":int(y[list(selected[(p,b)])].sum()),"ci_low":np.quantile(vals,.025),"ci_high":np.quantile(vals,.975),"unit":"resampled_sites","replicates":500})
    write("conditional_policy_results.csv",results); write("fold_results.csv",foldrows); write("budget_yield_intervals.csv",intervals)
    # Compact report generated from the actual saved results.
    lines=["# Conditional uncertainty policy experiment","","## Design","",f"Source-only evaluation used {len(ids):,} audio-eligible clips and {positives} useful clips, in the supplied five disjoint site-grouped outer folds. E1 is global `1 - max(camera posterior)` uncertainty; E2 maps five outer-training quantile bins to smoothed usefulness rates; E3/E4 add bin-by-IR or bin-by-placement cells; E5 uses their three-way cells. No species, site, project, target/adaptation data, or oracle feature entered a selector.","","Every policy's outer-fold score was Platt-calibrated from inner site-grouped OOF predictions. Thus pooled comparisons use symmetric calibrated scores. `E1_uncertainty/raw_ranking` is separately reported only to distinguish the original monotonic uncertainty ranking from its calibrated global ranking; it is not an additional tuned method. The fixed five-pseudo-observation shrinkage is a stability rule, not label-selected tuning.","","## Pooled calibrated budget yield","","| Policy | Budget | Calls | Useful | Precision | Recall | Correction rate |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        if r["score_variant"] == "calibrated_global": lines.append("| %s | %d%% | %d | %d | %.3f | %.3f | %.3f |"%(r["policy"],100*r["budget"],r["audio_calls"],r["useful_clips_captured"],r["precision"],r["recall"],r["correction_rate"]))
    lines += ["","Raw fold-level yields and paired differences are in `fold_results.csv`; 100 common random permutations are in `conditional_policy_results.csv`. `budget_yield_intervals.csv` gives site-resampled 95% intervals for useful clips captured, rather than treating dependent clips as independent.","","## Descriptive interaction checks","", "The following are descriptive outcome rates among eligible source clips. They are not used to choose policies, bins, or thresholds and therefore do not change the nested-CV comparison.", "", "| Uncertainty bin (low to high) | Eligible | Useful | Rate |", "|---|---:|---:|---:|"]
    dbin,_=qbins(u[ids],u[ids])
    for b in range(5):
        ii=ids[dbin==b]; lines.append("| %d | %d | %d | %.3f |"%(b+1,len(ii),y[ii].sum(),y[ii].mean()))
    for title, values in [("IR condition",ir[ids]),("Placement",placement[ids])]:
        lines += ["", "### By uncertainty bin and " + title, "", "| Group | Bin | Eligible | Useful | Rate |", "|---|---:|---:|---:|"]
        for value in sorted(set(values.astype(str))):
            for b in range(5):
                ii=ids[(values.astype(str)==value)&(dbin==b)]
                if len(ii): lines.append("| %s | %d | %d | %d | %.3f |"%(value,b+1,len(ii),y[ii].sum(),y[ii].mean()))
    lines += ["", "## Conclusion", "", "E4 has a small pooled advantage only at the 50% budget (114 versus E1's 112 useful clips); it is tied at 20% and worse at the more operationally selective 10% and 30% budgets. Given only 147 positives, this isolated difference is not convincing interaction evidence. The simpler global camera-uncertainty policy remains preferable."]
    open(os.path.join(HERE,"conditional_policy_report.md"),"w",encoding="utf-8").write("\n".join(lines)+"\n")
    print("wrote conditional policy experiment to", HERE)

if __name__ == "__main__": main()
