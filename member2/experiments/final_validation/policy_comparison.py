"""Policy metrics and paired site bootstrap for final nested OOF predictions."""
import math, numpy as np
from sklearn.metrics import f1_score, accuracy_score, log_loss
from nested_validation import BUDGETS, SEED, save

def _metrics(y, cp, fused, sel, delta):
    p=cp.copy(); p[sel]=fused[sel]
    return {'macro_f1':f1_score(y,p.argmax(1),average='macro',zero_division=0),'accuracy':accuracy_score(y,p.argmax(1)),
      'log_loss':log_loss(y,p,labels=np.arange(p.shape[1])),'multiclass_brier':float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2,axis=1))),
      'audio_calls':int(sel.sum()),'audio_acquisition_rate':float(sel.mean()),'total_incremental_benefit':float(delta[sel].sum()),
      'mean_incremental_benefit_per_call':float(delta[sel].mean()) if sel.any() else 0.,'harmful_acquisitions':int((delta[sel]<0).sum()),'beneficial_acquisitions':int((delta[sel]>0).sum()),'normalized_compute_proxy':float(sel.mean())}

def _choose(score,fold,budget,method):
    sel=np.zeros(len(score),bool)
    if method=='camera_only': return sel
    if method=='audio_always': return ~np.isnan(score)
    rng=np.random.default_rng(SEED)
    for f in sorted(set(fold)):
        ix=np.where(fold==f)[0]; k=math.ceil(budget*len(ix)); order=rng.permutation(ix) if method=='random' else ix[np.argsort(-score[ix],kind='stable')]
        sel[order[:k]]=True
    return sel

def compare(rows,classes,fold,groups,y,cp,ap,delta,scores,unc):
    fused=.95*cp+.05*ap; all_scores={**scores,'camera_uncertainty':unc}; records=[]; cache={}
    for budget in BUDGETS:
      for method in ['camera_only','audio_always','random','camera_uncertainty','binary_usefulness','shim_cost_sensitive','continuous_incremental','binary_incremental']:
        s=np.zeros(len(y)) if method in ('camera_only','audio_always','random') else all_scores[method]; sel=_choose(s,fold,budget,method); cache[(method,budget)]=sel
        records.append({'split':'source_nested_oof','policy':method,'budget':budget,**_metrics(y,cp,fused,sel,delta)})
    save('policy_comparison.csv',records)
    # Fixed-policy paired site bootstrap. It conditions on fitted nested OOF scores.
    sites=np.unique(groups); rng=np.random.default_rng(SEED); ci=[]
    refs=[('continuous_incremental','camera_uncertainty'),('binary_incremental','camera_uncertainty'),('continuous_incremental','binary_usefulness')]
    # The pre-specified primary operating point is 30%; fixed-budget results
    # above retain all budgets, while uncertainty focuses on that decision.
    for budget in (.30,):
      for a,b in refs:
        vals={k:[] for k in ('macro_f1','log_loss','total_incremental_benefit','mean_incremental_benefit_per_call','audio_acquisition_rate')}
        for _ in range(1000):
          ix=np.concatenate([np.where(groups==s)[0] for s in rng.choice(sites,len(sites),replace=True)])
          ma=_metrics(y[ix],cp[ix],fused[ix],cache[(a,budget)][ix],delta[ix]); mb=_metrics(y[ix],cp[ix],fused[ix],cache[(b,budget)][ix],delta[ix])
          for k in vals: vals[k].append(ma[k]-mb[k])
        for k,v in vals.items(): ci.append({'budget':budget,'comparison':a+' minus '+b,'metric':k,'ci_low':np.quantile(v,.025),'ci_high':np.quantile(v,.975),'bootstrap_replicates':1000,'n_sites':len(sites),'n_clips':len(y)})
    save('paired_site_bootstrap.csv',ci)
    # Predefined cost thresholds, never tuned on outer validation.
    cost=[]
    for method in ('continuous_incremental','binary_incremental','binary_usefulness','shim_cost_sensitive'):
      for c in (.001,.005,.01):
        # Regression is in log-benefit units. Classifiers use documented expected unit benefit=.10.
        threshold=c if method=='continuous_incremental' else c/.10
        sel=all_scores[method]>threshold
        cost.append({'policy':method,'normalized_audio_cost':c,'score_threshold':threshold,**_metrics(y,cp,fused,sel,delta)})
    save('cost_sensitivity.csv',cost); return records,ci
