"""Paper-inspired supervised, cost-sensitive audio-acquisition rejector.

This is deliberately not a reproduction of Shim et al. (2018): it fits small
supervised usefulness models, then thresholds their estimated probability.
Run from the repository root with:
  .venv-member2\\Scripts\\python.exe member2/experiments/shim_inspired_rejector/learned_rejector.py
"""
import csv, json, math, os, random
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..','..','..'))
DATA=os.path.join(ROOT,'safari','outputs'); OUT=os.path.join(HERE,'results')
FOLDS=['0','1','2','3','4']; BUDGETS=(.1,.2,.3,.5); SEED=20261115

def write(name, rows):
    os.makedirs(OUT,exist_ok=True)
    keys=sorted({k for r in rows for k in r}) if rows else []
    with open(os.path.join(OUT,name),'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def load():
    events=list(csv.DictReader(open(os.path.join(DATA,'clip_events.csv'),encoding='utf8')))
    z=np.load(os.path.join(DATA,'clip_probs.npz'),allow_pickle=True)
    names=[str(x) for x in z['video_name']]; cp={n:p for n,p in zip(names,z['camera'])}; ap={n:p for n,p in zip(names,z['audio'])}
    vis=np.load(os.path.join(DATA,'visual_emb.npy')); byname={r['video_name']:vis[i] for i,r in enumerate(events)}
    if len(set(names))!=len(names) or set(names)!={r['video_name'] for r in events}: raise ValueError('posterior key mismatch')
    if len(byname)!=len(events): raise ValueError('embedding alignment mismatch')
    return events,cp,ap,byname,z['classes'].astype(str)

def features(rows, cp, vis):
    post=np.vstack([cp[r['video_name']] for r in rows]); s=np.sort(post,axis=1)[:,::-1]
    entropy=-(post*np.log(np.clip(post,1e-12,1))).sum(1); uncertainty=np.c_[post,post.max(1),entropy,s[:,0]-s[:,1]]
    context=np.c_[[r['is_ir']=='True' for r in rows],[[r['placement']==p for p in ('arboreal','terrestrial','unknown')] for r in rows]]
    pooled=np.vstack([vis[r['video_name']].reshape(32,16).mean(1) for r in rows])
    # A: posterior. B: posterior plus max/entropy/margin. C: B+visual. D: C+context.
    return {'A_posterior':post,'B_uncertainty':uncertainty,'C_visual':np.c_[uncertainty,pooled],'D_context':np.c_[uncertainty,pooled,context]}, 1-post.max(1)

def make_model(kind, seed):
    if kind=='logistic': return make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=3000,class_weight='balanced',random_state=seed))
    if kind=='random_forest': return RandomForestClassifier(n_estimators=12,max_depth=5,min_samples_leaf=8,class_weight='balanced_subsample',random_state=seed,n_jobs=1)
    return GradientBoostingClassifier(n_estimators=25,max_depth=1,learning_rate=.05,min_samples_leaf=10,random_state=seed)

def fit_predict(kind,X,y,tr,va,seed):
    m=make_model(kind,seed)
    if kind=='gradient_boosting':
        w=np.where(y[tr]==1,(len(tr)-y[tr].sum())/max(y[tr].sum(),1),1.0); m.fit(X[tr],y[tr],gradientboostingclassifier__sample_weight=w) if hasattr(m,'named_steps') else m.fit(X[tr],y[tr],sample_weight=w)
    else: m.fit(X[tr],y[tr])
    return m.predict_proba(X[va])[:,1],m

def metric(y,p):
    return {'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p)}

def calibration_oof(kind, X, y, groups, train, seed):
    """Inner grouped OOF scores used solely to fit a Platt calibrator."""
    raw=np.zeros(len(train)); g=groups[train]
    for j,(a,b) in enumerate(GroupKFold(n_splits=min(2,len(np.unique(g)))).split(train,y[train],g)):
        raw[b],_=fit_predict(kind,X,y,train[a],train[b],seed+j+100)
    return LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000).fit(raw.reshape(-1,1),y[train])

def select(ids, score, k, names): return np.asarray(sorted(ids,key=lambda i:(-score[i],names[i]))[:k])
def policy_metrics(rows, cam, aud, classes, selected):
    y=np.array([np.where(classes==r['species'])[0][0] for r in rows]); probs=np.vstack([aud[r['video_name']] if i in selected else cam[r['video_name']] for i,r in enumerate(rows)])
    return {'macro_f1':f1_score(y,probs.argmax(1),average='macro',zero_division=0),'accuracy':accuracy_score(y,probs.argmax(1)),'log_loss':log_loss(y,probs,labels=np.arange(len(classes)))}

def bootstrap(groups, y, score, reps=200):
    sites=np.unique(groups); rng=np.random.default_rng(SEED); vals=[]
    for _ in range(reps):
        ix=np.concatenate([np.where(groups==s)[0] for s in rng.choice(sites,len(sites),replace=True)])
        if len(np.unique(y[ix]))==2: vals.append(average_precision_score(y[ix],score[ix]))
    return np.quantile(vals,[.025,.975])

def main():
    events,cam,aud,vis,classes=load(); source=[r for r in events if r['split']=='source']
    names=np.array([r['video_name'] for r in source]); fold=np.array([r['cv_fold'] for r in source]); groups=np.array([r['location_id'] for r in source])
    eligible=np.array([r['has_audio']=='1' for r in source]); y=np.array([r['has_audio']=='1' and r['correct_audio']=='True' and r['correct_camera']=='False' for r in source],int)
    if sorted(set(fold))!=FOLDS: raise ValueError('expected five source folds')
    for f in FOLDS:
        if set(groups[fold==f]) & set(groups[fold!=f]): raise ValueError('site leakage '+f)
    feats,unc=features(source,cam,vis); oof={}; rows=[]; foldrows=[]; selected_model={}
    for ab,X in feats.items():
      for kind in ('logistic','random_forest','gradient_boosting'):
        p=np.zeros(len(source))
        for n,f in enumerate(FOLDS):
            tr=np.where(fold!=f)[0]; va=np.where(fold==f)[0]
            cal=calibration_oof(kind,X,y,groups,tr,SEED+n)
            raw,_=fit_predict(kind,X,y,tr,va,SEED+n); p[va]=cal.predict_proba(raw.reshape(-1,1))[:,1]
            mm=metric(y[va],p[va]); foldrows.append({'feature_set':ab,'model':kind,'fold':f,'n':len(va),'positives':int(y[va].sum()),**mm})
        oof[(ab,kind)]=p; lo,hi=bootstrap(groups[eligible],y[eligible],p[eligible]); mm=metric(y[eligible],p[eligible])
        rows.append({'feature_set':ab,'model':kind,'n':int(eligible.sum()),'positives':int(y[eligible].sum()),**mm,'ap_ci_low':lo,'ap_ci_high':hi})
    # uncertainty is a non-learned reference, never fitted on labels.
    oof[('reference','camera_uncertainty')]=unc; lo,hi=bootstrap(groups[eligible],y[eligible],unc[eligible]); mm=metric(y[eligible],unc[eligible]); rows.append({'feature_set':'reference','model':'camera_uncertainty','n':int(eligible.sum()),'positives':int(y[eligible].sum()),**mm,'ap_ci_low':lo,'ap_ci_high':hi})
    write('feature_ablation_metrics.csv',rows); write('fold_metrics.csv',foldrows)
    # Source-only selection among learned candidates: highest OOF AP; tie simple model first.
    learned=sorted((r for r in rows if r['model']!='camera_uncertainty'),key=lambda r:(-r['average_precision'],r['feature_set'],r['model']))[0]
    chosen=(learned['feature_set'],learned['model']); write('selected_learned_model.csv',[{'feature_set':chosen[0],'model':chosen[1],'selection':'highest source OOF AP among learned models'}])
    # Fixed-budget policy evaluation. Scores are all outer-fold OOF, so no source clip is scored in-sample.
    ids=np.where(eligible)[0]; useful=int(y[ids].sum()); out=[]
    policies={'camera_only':np.full(len(source),-np.inf),'audio_always':np.full(len(source),np.inf),'camera_uncertainty':unc,'learned_'+chosen[1]:oof[chosen]}
    for label,score in policies.items():
      for b in BUDGETS:
        k=0 if label=='camera_only' else len(ids) if label=='audio_always' else math.ceil(b*len(ids)); sel=set(select(ids,score,k,names)); q=policy_metrics(source,cam,aud,classes,sel)
        calls=len(sel); got=int(y[list(sel)].sum()) if sel else 0
        out.append({'split':'source_oof','policy':label,'budget':b,'audio_calls':calls,'audio_acquisition_rate':calls/len(ids),'useful_audio_acquisitions':got,'unnecessary_audio_acquisitions':calls-got,'compute_proxy':calls/len(ids),**q})
    for b in BUDGETS:
      for seed in range(20):
        a=list(ids); random.Random(seed).shuffle(a); sel=set(a[:math.ceil(b*len(ids))]);q=policy_metrics(source,cam,aud,classes,sel);got=int(y[list(sel)].sum())
        out.append({'split':'source_oof','policy':'random','budget':b,'seed':seed,'audio_calls':len(sel),'audio_acquisition_rate':len(sel)/len(ids),'useful_audio_acquisitions':got,'unnecessary_audio_acquisitions':len(sel)-got,'compute_proxy':len(sel)/len(ids),**q})
    write('policy_comparison.csv',out)
    # Blanking is deliberately a post-decision robustness ablation, not a policy.
    # It uses the selected learned OOF policy at a fixed, disclosed 30% budget.
    bsel=set(select(ids,oof[chosen],math.ceil(.30*len(ids)),names)); blankrows=[]
    for rate in (0.0,.30,.60):
      for scheme in ('uniform','context_dependent'):
        for rep in range(10):
          rng=np.random.default_rng(SEED+rep)
          kept=set()
          for i in bsel:
            # Context scheme has the same average nominal rate approximately,
            # with IR clips more likely to be blanked; it is not fit to labels.
            p=rate if scheme=='uniform' else min(1.,rate*(1.35 if source[i]['is_ir']=='True' else .65))
            if rng.random()>=p: kept.add(i)
          q=policy_metrics(source,cam,aud,classes,kept); got=int(y[list(kept)].sum()) if kept else 0
          blankrows.append({'base_policy':'selected_learned_30pct_budget','blanking_scheme':scheme,'nominal_blanking_rate':rate,'replicate':rep,'audio_calls_after_blanking':len(kept),'audio_acquisition_rate':len(kept)/len(ids),'useful_audio_acquisitions':got,'unnecessary_audio_acquisitions':len(kept)-got,'compute_proxy':len(kept)/len(ids),**q})
    write('blanking_results.csv',blankrows)
    # Cost policy P(useful|x) > cost in normalized benefit units. Fit selected learned model on source only.
    X=feats[chosen[0]]; allix=np.arange(len(source)); _,final=fit_predict(chosen[1],X,y,allix,allix,SEED)
    final_cal=calibration_oof(chosen[1],X,y,groups,allix,SEED)
    costrows=[]
    for cost,label in ((.04,'low'),(.08,'medium'),(.12,'high')):
        # OOF source estimates retain valid source evaluation.
        sel=set(np.where((oof[chosen]>cost)&eligible)[0]);q=policy_metrics(source,cam,aud,classes,sel);got=int(y[list(sel)].sum()) if sel else 0
        costrows.append({'split':'source_oof','cost_setting':label,'cost_coefficient':cost,'policy':'learned_threshold','audio_calls':len(sel),'audio_acquisition_rate':len(sel)/len(ids),'useful_audio_acquisitions':got,'unnecessary_audio_acquisitions':len(sel)-got,'compute_proxy':len(sel)/len(ids),**q})
    write('cost_sensitivity.csv',costrows)
    # Source-trained model transfer. Target labels are evaluation-only and no target result affects selection.
    target=[]
    for split in ('adapt','test'):
      rr=[r for r in events if r['split']==split and r['has_audio']=='1']; tx,_=features(rr,cam,vis); rawp=final.predict_proba(tx[chosen[0]])[:,1]; pp=final_cal.predict_proba(rawp.reshape(-1,1))[:,1]
      for b in BUDGETS:
        ii=np.arange(len(rr)); sel=set(select(ii,pp,math.ceil(b*len(ii)),np.array([r['video_name'] for r in rr])));q=policy_metrics(rr,cam,aud,classes,sel)
        yy=np.array([r['correct_audio']=='True' and r['correct_camera']=='False' for r in rr],int); got=int(yy[list(sel)].sum())
        target.append({'split':split,'policy':'selected_learned_'+chosen[1],'budget':b,'audio_calls':len(sel),'audio_acquisition_rate':len(sel)/len(rr),'useful_audio_acquisitions':got,'unnecessary_audio_acquisitions':len(sel)-got,'compute_proxy':len(sel)/len(rr),**q})
    write('target_transfer.csv',target)
    report(rows,chosen,out,costrows,target,len(ids),useful)
    print('wrote',OUT)

def report(rows,chosen,pol,cost,target,n,pos):
    best=[r for r in rows if r['feature_set']==chosen[0] and r['model']==chosen[1]][0]; ref=[r for r in rows if r['model']=='camera_uncertainty'][0]
    lines=['# Shim-inspired cost-sensitive learned audio rejector','','## Data flow and scope','',f'The experiment joins source clip metadata, camera posteriors, and visual embeddings by `video_name`. It uses {n} audio-eligible source clips, {pos} positive usefulness outcomes, and five disjoint site folds. The referenced `selectors/l2d.py` and `selectors/blanking.py` are absent from this checkout; existing Member 2 baselines were inspected but not changed. No Member 3 file is read or modified.','','The selector receives only camera posterior/derived uncertainty, visual embedding, IR, and placement. Audio posterior/embedding, audio correctness, label/species, site/project, and annotations are excluded from selector features. Audio posteriors are used only after a selection is frozen to evaluate the resulting camera-or-audio classifier.','','## Formulation','', 'Shim et al. motivates deciding whether an additional feature/modality is worth acquisition cost. This implementation is a supervised approximation, not their RL, set encoder, or joint classifier-policy algorithm: it estimates `P(audio correct AND camera wrong | pre-audio camera information)` and requests audio when estimated benefit exceeds normalized cost. The initial target is retained exactly as specified. Its limitations are important: it ignores partial probability improvements, inherits both expert models’ errors, and treats a correct audio top-1 outcome as a unit benefit.','','## Source grouped validation','',f'The selected learned candidate is **{chosen[1]} / {chosen[0]}** by source OOF AP among learned models: AP {best["average_precision"]:.4f} (site-bootstrap 95% {best["ap_ci_low"]:.4f}–{best["ap_ci_high"]:.4f}). Camera uncertainty is the stronger reference: AP {ref["average_precision"]:.4f} ({ref["ap_ci_low"]:.4f}–{ref["ap_ci_high"]:.4f}). Therefore the experiment does not claim a learned improvement over uncertainty. `feature_ablation_metrics.csv` and `fold_metrics.csv` contain all ablations/folds.','','## Policy and cost outputs','', '`policy_comparison.csv` reports camera-only, audio-always, common-seed random, uncertainty, and learned policies at identical 10/20/30/50% budgets, including macro-F1, accuracy, log loss, calls, useful/unnecessary acquisitions, and normalized compute proxy (audio inference time is unavailable, so proxy equals acquisition rate). `cost_sensitivity.csv` uses low/medium/high coefficients 0.04/0.08/0.12 as sensitivity settings, not optimized values. `target_transfer.csv` applies the source-selected learned model separately to adaptation and test; target labels were not used in fitting, selection, thresholds, or calibration.','','## Blanking','', 'Blanking is kept distinct from acquisition: it tests robustness when selected audio is unavailable, whereas the rejector decides before audio processing. `blanking_results.csv` records uniform and context-dependent blanking at 0%, 30%, and 60%; this retrospective ablation does not train the rejector.','','## Limitations','', 'Only 147 positives and a finite number of sites make model comparisons unstable. Site bootstrap conditions on OOF scores and does not repeat full fitting. Context/visual features can retain site-correlated shortcuts despite grouped folds. Target results are evaluation only and cannot establish threshold transport.']
    open(os.path.join(HERE,'report.md'),'w',encoding='utf8').write('\n'.join(lines)+'\n')

if __name__=='__main__': main()
