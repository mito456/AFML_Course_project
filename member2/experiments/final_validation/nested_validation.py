"""Nested, site-grouped validation for the frozen Member 2 comparison.

All joins are by video_name.  Audio probabilities are retrospective outcomes
only: neither they nor labels/site identifiers enter selector features.
"""
import csv, json, math, os
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; DATA=ROOT/'safari'/'outputs'
FOLDS=['0','1','2','3','4']; SEED=20261115; BUDGETS=(.1,.2,.3,.5)

def save(name, rows):
    out=HERE/'results'; out.mkdir(exist_ok=True)
    keys=sorted({k for r in rows for k in r}) if rows else []
    with open(out/name,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,keys); w.writeheader(); w.writerows(rows)

def load():
    events=list(csv.DictReader(open(DATA/'clip_events.csv',encoding='utf8')))
    p=np.load(DATA/'clip_probs.npz',allow_pickle=True); names=[str(x) for x in p['video_name']]
    if len(names)!=len(set(names)) or set(names)!={r['video_name'] for r in events}: raise ValueError('video_name posterior join failed')
    cam=dict(zip(names,p['camera'])); aud=dict(zip(names,p['audio']))
    emb=np.load(DATA/'visual_emb.npy')
    # `visual_emb.npy` has no identifiers.  Its documented row manifest is
    # feat_index, so do not silently substitute clip_events row order.
    index=list(csv.DictReader(open(DATA/'feat_index.csv',encoding='utf8')))
    index_names=[r['video_name'] for r in index]; event_names=[r['video_name'] for r in events]
    if len(emb)!=len(index_names) or len(index_names)!=len(set(index_names)):
        raise ValueError('visual embedding / feat_index cardinality failed')
    if index_names != event_names:
        raise ValueError('feat_index order does not verify clip_events alignment')
    if set(index_names) != set(names):
        raise ValueError('feat_index keys do not match posterior keys')
    vis={name:emb[i] for i,name in enumerate(index_names)}
    src=[r for r in events if r['split']=='source' and r['has_audio']=='1']
    if any(r['prob_source']!='oof_site_cv' for r in src): raise ValueError('source expert predictions are not site OOF')
    fold=np.array([r['cv_fold'] for r in src]); group=np.array([r['location_id'] for r in src])
    if sorted(set(fold))!=FOLDS: raise ValueError('expected five supplied folds')
    for f in FOLDS:
        if set(group[fold==f]) & set(group[fold!=f]): raise ValueError('site overlap in fold '+f)
    classes=p['classes'].astype(str); lookup={x:i for i,x in enumerate(classes)}
    y=np.array([lookup[r['species']] for r in src]); cp=np.vstack([cam[r['video_name']] for r in src]); ap=np.vstack([aud[r['video_name']] for r in src])
    fused=.95*cp+.05*ap; delta=np.log(np.clip(fused[np.arange(len(y)),y],1e-12,1))-np.log(np.clip(cp[np.arange(len(y)),y],1e-12,1))
    useful=np.array([(r['correct_audio']=='True' and r['correct_camera']=='False') for r in src],int)
    return events,src,cam,aud,vis,classes,fold,group,y,cp,ap,delta,useful

def features(rows, cam, vis):
    p=np.vstack([cam[r['video_name']] for r in rows]); q=np.sort(p,axis=1)[:,::-1]
    info=np.c_[p,p.max(1),-(p*np.log(np.clip(p,1e-12,1))).sum(1),q[:,0]-q[:,1]]
    pooled=np.vstack([vis[r['video_name']].reshape(32,16).mean(1) for r in rows])
    context=np.array([[r['is_ir']=='True',r['placement']=='arboreal',r['placement']=='terrestrial',r['placement']=='unknown'] for r in rows],float)
    return {'posterior':p,'posterior_uncertainty':info,'visual':np.c_[info,pooled],'context':np.c_[info,pooled,context]}, info[:,-2]

def model(task, kind, seed):
    if task=='classification':
        if kind=='logistic': return make_pipeline(StandardScaler(),LogisticRegression(C=.2,class_weight='balanced',max_iter=200,solver='liblinear',random_state=seed))
        return RandomForestClassifier(n_estimators=20,max_depth=5,min_samples_leaf=8,class_weight='balanced_subsample',random_state=seed,n_jobs=1)
    if kind=='ridge': return make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    return RandomForestRegressor(n_estimators=20,max_depth=5,min_samples_leaf=8,random_state=seed,n_jobs=1)

def fit_score(task, kind, X, target, tr, va, seed):
    m=model(task,kind,seed); m.fit(X[tr],target[tr])
    return m.predict_proba(X[va])[:,1] if task=='classification' else m.predict(X[va])

def inner_scores(task, candidates, feats, target, groups, train, seed):
    # Two grouped inner folds retain site-held-out selection while keeping this
    # controlled confirmation pass within its fixed computational scope.
    ng=min(2,len(np.unique(groups[train]))); result={}
    for ci,(fs,kind,tag) in enumerate(candidates):
        pred=np.zeros(len(train)); X=feats[fs]
        for j,(a,b) in enumerate(GroupKFold(ng).split(train,target[train],groups[train])):
            pred[b]=fit_score(task,kind,X,target,train[a],train[b],seed+ci*31+j)
        result[(fs,kind,tag)]=pred
    return result

def select_candidate(task,candidates,feats,target,delta,groups,train,names,seed):
    scores=inner_scores(task,candidates,feats,target,groups,train,seed); choices=[]
    for c,p in scores.items():
        k=math.ceil(.30*len(train)); chosen=np.argsort(-p,kind='stable')[:k]
        # All selection happens here, using only outer-training clips.
        value=float(delta[train][chosen].sum()) if task=='regression' or c[2].startswith('delta') else float(target[train][chosen].sum())
        choices.append((value,c))
    return max(choices,key=lambda z:(z[0],-candidates.index(z[1])))[1],choices

def run_nested():
    events,rows,cam,aud,vis,classes,fold,groups,y,cp,ap,delta,useful=load(); feats,unc=features(rows,cam,vis); names=np.array([r['video_name'] for r in rows])
    # Deliberately low-capacity candidates: this is confirmation, not another
    # architecture search.  The four feature ablations are the prespecified grid.
    candidates_cls=[(f,'logistic','useful') for f in feats]
    candidates_reg=[(f,'ridge','delta') for f in feats]
    candidates_bin=[(f,'logistic','delta_gt_'+str(t)) for f in feats for t in (0,.002,.01)]
    scores={x:np.zeros(len(rows)) for x in ('binary_usefulness','shim_cost_sensitive','continuous_incremental','binary_incremental')}
    chosen_rows=[]
    for fi,f in enumerate(FOLDS):
        tr=np.where(fold!=f)[0]; va=np.where(fold==f)[0]
        # Shim shares binary usefulness's target, grid, and ranked decision;
        # fit it once and retain the alias only for backwards-compatible output.
        for label,task,cands,target in [('binary_usefulness','classification',candidates_cls,useful),('continuous_incremental','regression',candidates_reg,delta),('binary_incremental','classification',candidates_bin,None)]:
            if label=='binary_incremental':
                # Candidate-specific meaningful-benefit labels; select on inner realized delta.
                best=None; trials=[]
                for c in cands:
                    t=float(c[2].split('_')[-1]); targ=(delta>t).astype(int)
                    p=inner_scores('classification',[c],feats,targ,groups,tr,SEED+fi)[c]; sel=np.argsort(-p,kind='stable')[:math.ceil(.3*len(tr))]; val=float(delta[tr][sel].sum()); trials.append((val,c,targ))
                _,c,target=max(trials,key=lambda z:z[0]); choice=c
            else:
                choice,trials=select_candidate(task,cands,feats,target,delta,groups,tr,names,SEED+fi)
            fs,kind,tag=choice; scores[label][va]=fit_score(task,kind,feats[fs],target,tr,va,SEED+fi+500)
            chosen_rows.append({'outer_fold':f,'method':label,'feature_set':fs,'model':kind,'target':tag,'inner_selection':'30pct inner OOF realized benefit (usefulness count for A/Shim)'})
        scores['shim_cost_sensitive'][va]=scores['binary_usefulness'][va]
        chosen_rows.append({'outer_fold':f,'method':'shim_cost_sensitive','feature_set':'same_as_binary_usefulness','model':'same_as_binary_usefulness','target':'same_as_binary_usefulness','inner_selection':'operationally equivalent alias'})
    out=[]
    for method,score in scores.items():
        out += [{'video_name':names[i],'outer_fold':fold[i],'site':groups[i],'method':method,'score':score[i],'delta_95_05':delta[i],'binary_usefulness':useful[i]} for i in range(len(rows))]
    out += [{'video_name':names[i],'outer_fold':fold[i],'site':groups[i],'method':'camera_uncertainty','score':unc[i],'delta_95_05':delta[i],'binary_usefulness':useful[i]} for i in range(len(rows))]
    save('nested_oof_scores.csv',out); save('inner_selections.csv',chosen_rows)
    return rows,cam,aud,classes,fold,groups,y,cp,ap,delta,useful,scores,unc,chosen_rows
