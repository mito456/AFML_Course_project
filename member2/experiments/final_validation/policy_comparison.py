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

def _bootstrap_metrics(y, cp, fused, sel, delta):
    """Metric subset used in the bootstrap; avoids repeatedly invoking sklearn."""
    p=cp.copy(); p[sel]=fused[sel]
    pred=p.argmax(1); nclass=p.shape[1]
    cm=np.bincount(y*nclass+pred,minlength=nclass*nclass).reshape(nclass,nclass)
    den=2*np.diag(cm)+cm.sum(0)-np.diag(cm)+cm.sum(1)-np.diag(cm)
    f1=np.divide(2*np.diag(cm),den,out=np.zeros(nclass,float),where=den!=0).mean()
    ll=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()
    total=float(delta[sel].sum()); calls=int(sel.sum())
    return {'macro_f1':f1,'log_loss':ll,'total_incremental_benefit':total,
            'mean_incremental_benefit_per_call':total/calls if calls else 0.,
            'audio_acquisition_rate':float(sel.mean())}

def compare(rows,classes,fold,groups,y,cp,ap,delta,scores,unc):
    fused=.95*cp+.05*ap; all_scores={**scores,'camera_uncertainty':unc}; records=[]; cache={}
    for budget in BUDGETS:
      for method in ['camera_only','audio_always','random','camera_uncertainty','binary_usefulness','shim_cost_sensitive','continuous_incremental','binary_incremental','oracle']:
        # Oracle is retrospective-only: realized delta ranks clips.  As with
        # every existing policy, this is exact per-fold ceil(budget*n) selection,
        # including negative deltas if necessary.
        s=np.zeros(len(y)) if method in ('camera_only','audio_always','random') else (delta if method=='oracle' else all_scores[method]); sel=_choose(s,fold,budget,method); cache[(method,budget)]=sel
        records.append({'split':'source_nested_oof','policy':method,'budget':budget,**_metrics(y,cp,fused,sel,delta)})
    save('policy_comparison.csv',records)
    # Fixed-policy paired site bootstrap. It conditions on fitted nested OOF scores.
    sites=np.unique(groups); rng=np.random.default_rng(SEED); ci=[]
    learned=('binary_usefulness','shim_cost_sensitive','continuous_incremental','binary_incremental')
    refs=[('continuous_incremental','camera_uncertainty'),('binary_incremental','camera_uncertainty'),('continuous_incremental','binary_incremental')]
    refs += [(m,'random') for m in learned] + [(m,'camera_only') for m in learned]
    # The pre-specified primary operating point is 30%; fixed-budget results
    # above retain all budgets, while uncertainty focuses on that decision.
    for budget in (.30,):
      # Cache additive per-site sufficient statistics.  This keeps 2,000
      # whole-site resamples exact while avoiding repeated clip-level sklearn.
      stat={}
      for method in {m for pair in refs for m in pair}:
        cms=[]; nll=[]; benefit=[]; calls=[]; counts=[]; sel=cache[(method,budget)]
        for site in sites:
          ix=np.where(groups==site)[0]; p=cp[ix].copy(); p[sel[ix]]=fused[ix][sel[ix]]; pred=p.argmax(1); c=p.shape[1]
          cms.append(np.bincount(y[ix]*c+pred,minlength=c*c).reshape(c,c))
          nll.append(-np.log(np.clip(p[np.arange(len(ix)),y[ix]],1e-12,1)).sum())
          benefit.append(delta[ix][sel[ix]].sum()); calls.append(sel[ix].sum()); counts.append(len(ix))
        stat[method]=(np.asarray(cms),np.asarray(nll),np.asarray(benefit),np.asarray(calls),np.asarray(counts))
      def boot_metric(method, picked):
        cm,nll,benefit,calls,counts=stat[method]; cm=cm[picked].sum(0); n=counts[picked].sum(); call=calls[picked].sum(); den=2*np.diag(cm)+cm.sum(0)-np.diag(cm)+cm.sum(1)-np.diag(cm)
        return {'macro_f1':np.divide(2*np.diag(cm),den,out=np.zeros(len(den),float),where=den!=0).mean(), 'log_loss':nll[picked].sum()/n, 'total_incremental_benefit':benefit[picked].sum(), 'mean_incremental_benefit_per_call':benefit[picked].sum()/call if call else 0., 'audio_acquisition_rate':call/n}
      for a,b in refs:
        vals={k:[] for k in ('macro_f1','log_loss','total_incremental_benefit','mean_incremental_benefit_per_call','audio_acquisition_rate')}
        for _ in range(2000):
          picked=rng.integers(len(sites),size=len(sites))
          ma=boot_metric(a,picked); mb=boot_metric(b,picked)
          for k in vals: vals[k].append(ma[k]-mb[k])
        for k,v in vals.items(): ci.append({'budget':budget,'comparison':a+' minus '+b,'metric':k,'ci_low':np.quantile(v,.025),'ci_high':np.quantile(v,.975),'bootstrap_replicates':2000,'valid_bootstrap_samples':len(v),'bootstrap_seed':SEED,'resampling_unit':'site','n_sites':len(sites),'n_clips':len(y)})
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
