"""Leakage-controlled incremental predictive-value audio acquisition experiment.

Run from the repository root:
  .venv-member2\\Scripts\\python.exe member2/experiments/incremental_audio_value/incremental_value_rejector.py

The source camera and audio posteriors are explicitly checked to be site-OOF
(`prob_source == oof_site_cv`).  Audio is used only to construct/evaluate a
retrospective target, never as a rejector feature.
"""
import csv, math, os, random
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, log_loss, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..','..','..'))
DATA=os.path.join(ROOT,'safari','outputs'); OUT=os.path.join(HERE,'results')
FOLDS=['0','1','2','3','4']; BUDGETS=(.10,.20,.30,.50); SEED=20261115
EPS=(0.002,0.01) # pre-specified log-loss improvements: tiny and practical

def write(name, rows):
 os.makedirs(OUT,exist_ok=True); keys=sorted({k for r in rows for k in r}) if rows else []
 with open(os.path.join(OUT,name),'w',newline='',encoding='utf8') as h: csv.DictWriter(h,fieldnames=keys).writeheader(); h.seek(0,2)
 with open(os.path.join(OUT,name),'a',newline='',encoding='utf8') as h: csv.DictWriter(h,fieldnames=keys).writerows(rows)

def load():
 events=list(csv.DictReader(open(os.path.join(DATA,'clip_events.csv'),encoding='utf8')))
 z=np.load(os.path.join(DATA,'clip_probs.npz'),allow_pickle=True); names=[str(x) for x in z['video_name']]
 if len(names)!=len(set(names)) or set(names)!={r['video_name'] for r in events}: raise ValueError('posterior/event key mismatch')
 cp={n:p for n,p in zip(names,z['camera'])}; ap={n:p for n,p in zip(names,z['audio'])}
 visual=np.load(os.path.join(DATA,'visual_emb.npy')); vis={r['video_name']:visual[i] for i,r in enumerate(events)}
 if len(vis)!=len(events): raise ValueError('visual alignment mismatch')
 source=[r for r in events if r['split']=='source']
 if any(r['prob_source']!='oof_site_cv' for r in source): raise ValueError('source posterior is not OOF')
 return events,cp,ap,vis,z['classes'].astype(str)

def feat(rows,cp,vis):
 p=np.vstack([cp[r['video_name']] for r in rows]); s=np.sort(p,axis=1)[:,::-1]
 u=np.c_[p,p.max(1),-(p*np.log(np.clip(p,1e-12,1))).sum(1),s[:,0]-s[:,1]]
 v=np.vstack([vis[r['video_name']].reshape(32,16).mean(1) for r in rows])
 c=np.c_[[r['is_ir']=='True' for r in rows],[[r['placement']==x for x in ('arboreal','terrestrial','unknown')] for r in rows]]
 return {'A_posterior':p,'B_uncertainty':u,'C_visual':np.c_[u,v],'D_context':np.c_[u,v,c]}, 1-p.max(1)

def make(kind):
 if kind=='ridge': return make_pipeline(StandardScaler(),Ridge(alpha=20.))
 if kind=='rf_reg': return RandomForestRegressor(n_estimators=100,max_depth=5,min_samples_leaf=10,max_features=.7,random_state=SEED,n_jobs=1)
 if kind=='gb_reg': return GradientBoostingRegressor(n_estimators=40,max_depth=1,learning_rate=.04,min_samples_leaf=12,loss='huber',random_state=SEED)
 if kind=='logistic': return make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=3000,class_weight='balanced',random_state=SEED))
 if kind=='rf_clf': return RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_leaf=10,class_weight='balanced_subsample',random_state=SEED,n_jobs=1)
 return GradientBoostingClassifier(n_estimators=40,max_depth=1,learning_rate=.04,min_samples_leaf=12,random_state=SEED)
def predict(kind,X,y,tr,va):
 m=make(kind); m.fit(X[tr],y[tr]); return (m.predict(X[va]) if kind.endswith('reg') or kind=='ridge' else m.predict_proba(X[va])[:,1]),m
def ranking(ids,score,k,names): return np.array(sorted(ids,key=lambda i:(-score[i],names[i]))[:k])
def posterior(rows,cam,aud,sel,w): return np.vstack([(w*cam[r['video_name']]+(1-w)*aud[r['video_name']]) if i in sel else cam[r['video_name']] for i,r in enumerate(rows)])
def outcome(rows,cam,aud,classes,sel,delta,w):
 y=np.array([np.where(classes==r['species'])[0][0] for r in rows]); p=posterior(rows,cam,aud,sel,w)
 calls=len(sel); benefit=float(delta[list(sel)].sum()) if calls else 0.
 return {'macro_f1':f1_score(y,p.argmax(1),average='macro',zero_division=0),'accuracy':accuracy_score(y,p.argmax(1)),'log_loss':log_loss(y,p,labels=np.arange(len(classes))),'audio_calls':calls,'audio_acquisition_rate':calls/sum(r['has_audio']=='1' for r in rows),'total_incremental_benefit':benefit,'mean_benefit_per_call':benefit/calls if calls else 0.,'compute_proxy':calls/sum(r['has_audio']=='1' for r in rows)}
def bootstrap(rows,groups,cam,aud,classes,sel_a,sel_b,delta,w,reps=1000):
 # Paired site bootstrap. Precompute each policy's per-clip outcome; this is
 # mathematically identical to refitting no models in a resample, but avoids
 # repeatedly materialising posterior dictionaries 8,000 times.
 sites=np.unique(groups); rng=np.random.default_rng(SEED); vals=defaultdict(list)
 y=np.array([np.where(classes==r['species'])[0][0] for r in rows]); pc=np.vstack([cam[r['video_name']] for r in rows]); pa=np.vstack([aud[r['video_name']] for r in rows]); fused=w*pc+(1-w)*pa
 def prepared(sel):
  take=np.array([i in sel for i in range(len(rows))]); p=np.where(take[:,None],fused,pc)
  return take,p.argmax(1),-np.log(np.clip(p[np.arange(len(rows)),y],1e-12,1))
 ta,pa_pred,lla=prepared(sel_a); tb,pb_pred,llb=prepared(sel_b); nclass=len(classes)
 def mf1(truth,pred):
  cm=np.bincount(nclass*truth+pred,minlength=nclass*nclass).reshape(nclass,nclass); tp=np.diag(cm); den=cm.sum(0)+cm.sum(1)
  return float(np.mean(np.divide(2*tp,den,out=np.zeros(nclass,float),where=den>0)))
 for _ in range(reps):
  ix=np.concatenate([np.where(groups==s)[0] for s in rng.choice(sites,len(sites),replace=True)])
  for key,a,b in [('macro_f1',mf1(y[ix],pa_pred[ix]),mf1(y[ix],pb_pred[ix])),('log_loss',lla[ix].mean(),llb[ix].mean()),('total_incremental_benefit',np.sum(delta[ix]*ta[ix]),np.sum(delta[ix]*tb[ix])),('mean_benefit_per_call',np.sum(delta[ix]*ta[ix])/max(ta[ix].sum(),1),np.sum(delta[ix]*tb[ix])/max(tb[ix].sum(),1))]: vals[key].append(a-b)
 return [{'metric':k,'ci_low':np.quantile(v,.025),'ci_high':np.quantile(v,.975),'bootstrap_replicates':reps} for k,v in vals.items()]

def main():
 events,cam,aud,vis,classes=load(); source=[r for r in events if r['split']=='source']; n=len(source)
 names=np.array([r['video_name'] for r in source]); folds=np.array([r['cv_fold'] for r in source]); groups=np.array([r['location_id'] for r in source]); eligible=np.array([r['has_audio']=='1' for r in source])
 if sorted(set(folds))!=FOLDS: raise ValueError('expected five folds')
 for f in FOLDS:
  if set(groups[folds==f])&set(groups[folds!=f]): raise ValueError('site leakage')
 y=np.array([np.where(classes==r['species'])[0][0] for r in source]); pc=np.vstack([cam[x] for x in names]); pa=np.vstack([aud[x] for x in names])
 # log loss difference simplifies to log(p_fused[y]/p_camera[y]); clipping is numerical protection only.
 def delta(weight): return np.log(np.clip((weight*pc+(1-weight)*pa)[np.arange(n),y],1e-12,1))-np.log(np.clip(pc[np.arange(n),y],1e-12,1))
 d=delta(.95); feats,unc=feat(source,cam,vis); target_rows=[]
 for weight in (.95,.90):
  dd=delta(weight); target_rows.append({'fusion_camera_weight':weight,'n':n,'mean_delta':dd.mean(),'median_delta':np.median(dd),'sd_delta':dd.std(),'positive_rate':(dd>0).mean(),'negative_rate':(dd<0).mean(),'zero_rate':(dd==0).mean()})
 write('incremental_value_target_summary.csv',target_rows)
 variants=[('continuous',d),('positive', (d>0).astype(int)),*[(f'epsilon_{e:g}',(d>e).astype(int)) for e in EPS]]
 oof={}; abrows=[]; foldrows=[]
 for target,yy in variants:
  kinds=('ridge','rf_reg','gb_reg') if target=='continuous' else ('logistic','rf_clf','gb_clf')
  for fs,X in feats.items():
   for kind in kinds:
    score=np.zeros(n)
    for f in FOLDS:
     tr=np.where(folds!=f)[0]; va=np.where(folds==f)[0]; score[va],_=predict(kind,X,yy,tr,va)
     foldrows.append({'target':target,'feature_set':fs,'model':kind,'fold':f,'n':len(va),'target_mean':yy[va].mean(),'ranking_ap_positive_delta':average_precision_score((d[va]>0).astype(int),score[va]),'mean_delta_top30':d[ranking(va,score,.3 and math.ceil(.3*len(va)),names)].mean()})
    oof[(target,fs,kind)]=score
    # common operational criterion: true source-OOF delta captured at 30%, not target labels.
    ids=np.where(eligible)[0]; sel=ranking(ids,score,math.ceil(.3*len(ids)),names)
    abrows.append({'target':target,'feature_set':fs,'model':kind,'source_oof_top30_total_delta':d[sel].sum(),'source_oof_top30_mean_delta':d[sel].mean(),'ranking_ap_positive_delta':average_precision_score((d[eligible]>0).astype(int),score[eligible])})
 write('feature_ablation_metrics.csv',abrows); write('fold_metrics.csv',foldrows)
 # Selection occurs only on source OOF operational benefit.  Simple lexical tie-break is pre-declared.
 candidates=sorted(abrows,key=lambda r:(-r['source_oof_top30_total_delta'],r['target'],r['feature_set'],r['model']))
 chosen=candidates[0]; key=(chosen['target'],chosen['feature_set'],chosen['model']); write('selected_model.csv',[{**chosen,'selection':'highest source-OOF total incremental benefit at pre-specified 30% budget'}])
 ids=np.where(eligible)[0]; legacy={r['video_name']:float(r['predicted_probability']) for r in csv.DictReader(open(os.path.join(ROOT,'member2','availability.csv'),encoding='utf8'))}
 scores={'camera_uncertainty':unc,'existing_binary_usefulness':np.array([legacy[x] for x in names]),'continuous_incremental':oof[key] if chosen['target']=='continuous' else np.zeros(n),'binary_incremental':oof[key] if chosen['target']!='continuous' else np.zeros(n)}
 # Include best of each main formulation even if it was not the global selected candidate.
 for target,label in [('continuous','continuous_incremental'),('positive','binary_incremental')]:
  best=sorted([r for r in abrows if r['target']==target],key=lambda r:-r['source_oof_top30_total_delta'])[0]; scores[label]=oof[(target,best['feature_set'],best['model'])]
 policies=[]; cis=[]
 for w in (.95,.90):
  dd=delta(w)
  for budget in BUDGETS:
   base={}
   for label,sc in scores.items(): base[label]=set(ranking(ids,sc,math.ceil(budget*len(ids)),names))
   base['camera_only']=set(); base['audio_always']=set(ids)
   for label,sel in base.items(): policies.append({'split':'source_oof','fusion_camera_weight':w,'policy':label,'budget':budget,**outcome(source,cam,aud,classes,sel,dd,w)})
   for seed in range(20):
    rr=list(ids); random.Random(seed).shuffle(rr); sel=set(rr[:math.ceil(budget*len(ids))]); policies.append({'split':'source_oof','fusion_camera_weight':w,'policy':'random','budget':budget,'seed':seed,**outcome(source,cam,aud,classes,sel,dd,w)})
   if w==.95:
    for label in ('continuous_incremental','binary_incremental'):
     for r in bootstrap(source,groups,cam,aud,classes,base[label],base['camera_uncertainty'],dd,w): cis.append({'split':'source_oof','policy':label,'reference':'camera_uncertainty','budget':budget,**r})
 write('policy_comparison.csv',policies); write('site_bootstrap_ci.csv',cis)
 # Cost threshold applies expected value. Binary probability threshold is cost / pre-specified unit benefit (0.10).
 cost=[]
 for cost_value,cost_name in ((0.001,'low'),(.005,'medium'),(.01,'high')):
  for label in ('continuous_incremental','binary_incremental'):
   sc=scores[label]; sel=set(np.where(eligible & ((sc>cost_value) if label=='continuous_incremental' else (sc>cost_value/.10)))[0])
   cost.append({'split':'source_oof','policy':label,'cost_setting':cost_name,'cost':cost_value,'decision_rule':'predicted_delta > cost' if label=='continuous_incremental' else 'P(delta>0) > cost/0.10',**outcome(source,cam,aud,classes,sel,d,.95)})
 write('cost_sensitivity.csv',cost)
 # final source-only fit, then untouched target adaptation/test evaluation.
 X=feats[chosen['feature_set']]; yy=dict(variants)[chosen['target']]; _,final=predict(chosen['model'],X,yy,np.arange(n),np.arange(n))
 transfer=[]
 for split in ('adapt','test'):
  rr=[r for r in events if r['split']==split and r['has_audio']=='1']; tx,_=feat(rr,cam,vis); ss=final.predict(tx[chosen['feature_set']]) if chosen['model'].endswith('reg') or chosen['model']=='ridge' else final.predict_proba(tx[chosen['feature_set']])[:,1]
  ty=np.array([np.where(classes==r['species'])[0][0] for r in rr]); cc=np.vstack([cam[r['video_name']] for r in rr]); aa=np.vstack([aud[r['video_name']] for r in rr]); td=np.log(np.clip((.95*cc+.05*aa)[np.arange(len(rr)),ty],1e-12,1))-np.log(np.clip(cc[np.arange(len(rr)),ty],1e-12,1)); ii=np.arange(len(rr)); nn=np.array([r['video_name'] for r in rr])
  for b in BUDGETS:
   sel=set(ranking(ii,ss,math.ceil(b*len(ii)),nn)); transfer.append({'split':split,'policy':'source_selected_'+chosen['target'],'budget':b,**outcome(rr,cam,aud,classes,sel,td,.95)})
 write('target_transfer.csv',transfer); report(chosen,target_rows,abrows,policies,cis,transfer)
 print('wrote',OUT)

def report(chosen,summary,abrows,policies,cis,transfer):
 ref=[x for x in policies if x.get('fusion_camera_weight')==.95 and x['policy']=='camera_uncertainty' and x['budget']==.3][0]
 cont=[x for x in policies if x.get('fusion_camera_weight')==.95 and x['policy']=='continuous_incremental' and x['budget']==.3][0]
 binary=[x for x in policies if x.get('fusion_camera_weight')==.95 and x['policy']=='binary_incremental' and x['budget']==.3][0]
 lines=['# Incremental predictive value audio rejector','','## Method and provenance','','Binary top-1 usefulness (`audio correct AND camera wrong`) misses cases where audio increases the correct-class probability without flipping top-1. This experiment therefore defines `delta = log p_fused(y) - log p_camera(y)`, with a fixed 95/5 camera/audio posterior fusion. Positive delta means lower multiclass log loss after acquiring audio. Source `clip_events.prob_source` was checked to be `oof_site_cv` for every source row, and the supplied five site folds have no site overlap. Thus delta is constructed from OOF expert posteriors, not in-sample predictions.','','The rejector inputs are camera posterior/uncertainty, fixed visual pooling, IR, and placement only. It never receives audio outputs, labels/species, sites/projects, annotations, or delta. 90/10 is a pre-specified fusion sensitivity analysis, not a tuned weight.','','## Targets and validation','','Continuous regression predicts expected delta. Binary variants predict `delta > 0` and the pre-specified meaningful thresholds `delta > 0.002` and `delta > 0.01`; those thresholds were chosen as small and practical log-loss changes, not selected on target data. All fitting is outer site-OOF; the source selection rule is greatest OOF total true delta at a fixed pre-specified 30% budget. Results are noisy because delta is the difference of two imperfect OOF models.','','| Fusion camera weight | Mean delta | Median | SD | Positive rate |','|---:|---:|---:|---:|---:|']
 for x in summary: lines.append(f"| {x['fusion_camera_weight']:.2f} | {x['mean_delta']:.5f} | {x['median_delta']:.5f} | {x['sd_delta']:.5f} | {x['positive_rate']:.2%} |")
 lines+=['','## Primary source-OOF comparison (30% budget, 95/5 fusion)','','| Policy | Macro-F1 | Log loss | Total delta | Mean delta/call |','|---|---:|---:|---:|---:|']
 for label,x in [('camera uncertainty',ref),('continuous incremental',cont),('binary incremental',binary)]: lines.append(f"| {label} | {x['macro_f1']:.4f} | {x['log_loss']:.4f} | {x['total_incremental_benefit']:.4f} | {x['mean_benefit_per_call']:.5f} |")
 lines+=['','`feature_ablation_metrics.csv` and `fold_metrics.csv` contain every model/ablation/fold. `policy_comparison.csv` includes camera-only, audio-always, common-seed random, uncertainty, the existing binary rejector, and both incremental approaches at matching 10/20/30/50% budgets. `site_bootstrap_ci.csv` has 1,000 paired site-bootstrap intervals versus uncertainty for macro-F1, log loss, total benefit, and benefit/call. CIs condition on the fitted OOF scores and do not erase model-selection uncertainty.','','## Cost and target transfer','','`cost_sensitivity.csv` uses pre-specified normalized costs 0.001/0.005/0.01. Continuous acquisition requires predicted delta above cost; binary acquisition requires estimated positive-benefit probability above cost divided by a disclosed 0.10 unit-benefit scale. `target_transfer.csv` applies only the source-selected model to adaptation and test, separately; target labels did not influence features, selection, fitting, calibration, cost, or thresholds.','','## Interpretation and limits','','Do not infer superiority from a point estimate: use the paired site intervals and fold variation. Improvements in log loss need not improve macro-F1 because delta rewards calibrated correct-class probability even without a top-1 change. Visual/context features can retain site-correlated shortcuts despite grouped validation; expert posterior errors and a fixed fusion rule also propagate into delta. Small target populations and post-hoc comparison of many candidates further limit any domain-generalization claim.']
 open(os.path.join(HERE,'report.md'),'w',encoding='utf8').write('\n'.join(lines)+'\n')
 open(os.path.join(HERE,'methodology.md'),'w',encoding='utf8').write('''# Methodology\n\nThe experiment reads only existing source OOF camera/audio posteriors (`prob_source=oof_site_cv`) to make the retrospective incremental log-loss target. Rejector inputs are strictly pre-audio: camera posterior and deterministic uncertainty summaries, visual embedding, IR, and placement. Five supplied disjoint site folds provide source OOF scores; all preprocessing/model fits occur inside each outer training partition. Models are deliberately shallow/regularized. Audio posterior is used only in target construction and policy evaluation.\n\nThe primary decision uses 95/5 camera/audio fusion; 90/10 is sensitivity-only. Model/target selection is source-only by OOF total delta at a pre-specified 30% budget. Target adaptation/test are final source-fit evaluations only. Site-level paired bootstrap resamples locations, never individual clips.\n''')
 open(os.path.join(HERE,'README.md'),'w',encoding='utf8').write('''# Incremental audio value experiment\n\nRun `incremental_value_rejector.py` from the repository root with the Member 2 virtual environment. It writes all reproducible tables to `results/`, plus the report and methodology. Wrapper scripts run the same consolidated pipeline; use the main entry point once to avoid redundant reruns.\n''')
if __name__=='__main__': main()
