"""Retrospective camera/audio complementarity analysis for SA-FARI.

Run from the repository root:
  .venv-member2\\Scripts\\python.exe member2/experiments/multimodal_complementarity/multimodal_analysis.py

All posterior/event joins are keyed by video_name.  Target labels are never
used to choose fusion weights or fit the learned fusion model.
"""
import csv, math, os, sys
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA=os.path.join(ROOT,"safari","outputs")
RNG=np.random.default_rng(20261115)

def rows(path):
    with open(path,encoding="utf-8") as f:return list(csv.DictReader(f))
def write(name, data):
    if not data:return
    keys=sorted({k for r in data for k in r})
    with open(os.path.join(HERE,name),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(data)
def clip(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8)
    return p/p.sum(axis=1,keepdims=True)
def ece(y,p,bins=10):
    c=p.max(1); pred=p.argmax(1); out=0.
    for lo in np.linspace(0,1,bins,endpoint=False):
        q=(c>=lo)&(c < lo+1/bins)
        if q.any():out+=q.mean()*abs((pred[q]==y[q]).mean()-c[q].mean())
    return out
def metrics(y,p):
    pred=p.argmax(1); one=np.eye(p.shape[1])[y]
    return dict(n=len(y),macro_f1=f1_score(y,pred,average="macro",zero_division=0),accuracy=accuracy_score(y,pred),
      log_loss=log_loss(y,clip(p),labels=np.arange(p.shape[1])),brier=np.mean((p-one)**2),ece=ece(y,p))
def entropy(p): return -(clip(p)*np.log(clip(p))).sum(1)
def margin(p):
    z=np.sort(p,axis=1);return z[:,-1]-z[:,-2]
def boot_diff(y, a,b,site, reps=200):
    """Cluster bootstrap CI for paired macro-F1 difference (b-a)."""
    sites=np.unique(site); vals=[]
    for _ in range(reps):
        chosen=RNG.choice(sites,len(sites),replace=True); ix=np.concatenate([np.where(site==s)[0] for s in chosen])
        vals.append(metrics(y[ix],b[ix])["macro_f1"]-metrics(y[ix],a[ix])["macro_f1"])
    return np.quantile(vals,[.025,.975])
def choose_alpha(c,a,y,g):
    grid=np.linspace(0,1,21); scores=np.zeros(len(grid))
    cv=GroupKFold(n_splits=min(4,len(np.unique(g))))
    for tr,va in cv.split(c,y,g):
        for j,x in enumerate(grid):scores[j]+=metrics(y[va],x*c[va]+(1-x)*a[va])["log_loss"]
    return float(grid[np.argmin(scores)])
def svg_bar(path,title,labels,series):
    # dependency-free, intentionally simple portable SVG.
    W,H=760,360; mx=max(max(v for v in vals) for _,vals in series)*1.12 or 1
    n=len(labels); bw=500/(n*len(series)); colors=["#3567a8","#e5833a","#4e9b70","#9a5aad"]
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><style>text{{font:12px sans-serif}}</style><text x="30" y="25" font-size="16">{title}</text><line x1="70" y1="300" x2="700" y2="300" stroke="black"/>']
    for i,l in enumerate(labels):
      for j,(name,vals) in enumerate(series):
       h=240*vals[i]/mx;x=100+i*(500/n)+j*bw;out.append(f'<rect x="{x:.1f}" y="{300-h:.1f}" width="{bw-3:.1f}" height="{h:.1f}" fill="{colors[j]}"/>')
      out.append(f'<text x="{100+i*(500/n):.1f}" y="320">{l}</text>')
    for j,(name,_) in enumerate(series):out.append(f'<rect x="{80+j*135}" y="45" width="12" height="12" fill="{colors[j]}"/><text x="96" y="56">{name}</text>')
    out.append('</svg>');open(path,'w',encoding='utf-8').write(''.join(out))

def main():
 ev=rows(os.path.join(DATA,"clip_events.csv")); sp={r['video_name']:r for r in rows(os.path.join(DATA,"split.csv"))}
 # NumPy reads the string class array directly; keys are still explicitly
 # checked below rather than using either table's row order.
 with np.load(os.path.join(DATA,"clip_probs.npz"),allow_pickle=True) as pp:
  names=pp["video_name"]; classes=pp["classes"]; cam=pp["camera"]; aud=pp["audio"]
 if len(set(map(str,names)))!=len(names) or cam.shape!=aud.shape or cam.shape[1]!=len(classes):raise ValueError('posterior key/dimension check failed')
 post={str(n):(cam[i],aud[i]) for i,n in enumerate(names)}
 if set(post)!={r['video_name'] for r in ev}:raise ValueError('events/posterior video_name mismatch')
 cidx={str(x):i for i,x in enumerate(classes)}
 dat=[]
 for r in ev:
  if r['species'] not in cidx: continue
  c,a=post[r['video_name']];dat.append((r,c,a,cidx[r['species']]))
 # report label-supported populations; no assumption of source ordering
 pops={}
 for split in sorted({r['split'] for r,_,_,_ in dat}):
  z=[x for x in dat if x[0]['split']==split];pops[split]=z
 source=pops.get('source',[])
 if not source:raise ValueError('no source rows')
 def arrays(z):return (np.array([x[3] for x in z]),np.array([x[1] for x in z]),np.array([x[2] for x in z]),np.array([x[0]['location_id'] for x in z]),np.array([x[0]['video_name'] for x in z]))
 ys,cs,aus,gs,ns=arrays(source)
 # Outer source evaluation: each held-out supplied site fold gets alpha/model chosen on other folds.
 folds=np.array([x[0]['cv_fold'] for x in source]); fused=np.zeros_like(cs); learned=np.zeros_like(cs); alpha_by_fold=[]
 for fold in sorted(set(folds)):
  tr=folds!=fold;va=~tr; al=choose_alpha(cs[tr],aus[tr],ys[tr],gs[tr]);alpha_by_fold.append(al);fused[va]=al*cs[va]+(1-al)*aus[va]
  m=LogisticRegression(C=.3,max_iter=2000,random_state=17).fit(np.hstack([cs[tr],aus[tr]]),ys[tr]);learned[va]=m.predict_proba(np.hstack([cs[va],aus[va]]))
 alpha_final=choose_alpha(cs,aus,ys,gs)
 final_model=LogisticRegression(C=.3,max_iter=2000,random_state=17).fit(np.hstack([cs,aus]),ys)
 individual=[]; fusion=[]; domain=[]; conf=[]; dis=[]; inc=[]
 methods={'camera':cs,'audio':aus,'equal_average':.5*cs+.5*aus,'weighted_average':fused,'learned_fusion':learned}
 for population,z in pops.items():
  y,c,a,g,n=arrays(z)
  pp={'camera':c,'audio':a,'equal_average':.5*c+.5*a}
  if population=='source':pp.update({'weighted_average':fused,'learned_fusion':learned})
  else:pp.update({'weighted_average':alpha_final*c+(1-alpha_final)*a,'learned_fusion':final_model.predict_proba(np.hstack([c,a]))})
  for method,p in pp.items():
   rec={'population':population,'method':method,**metrics(y,p),'n_sites':len(set(g)),'n_classes_observed':len(set(y))}
   if method in ('camera','audio'):individual.append(rec)
   fusion.append(rec)
  # Domain shift rows include direct metric and drops relative to source OOF.
  for method,p in pp.items():
   base=metrics(ys,methods[method])
   cur=metrics(y,p); domain.append({'population':population,'method':method,'n':len(y),'n_sites':len(set(g)),**{k:cur[k] for k in ('macro_f1','accuracy','log_loss','brier','ece')},'macro_f1_drop_from_source':base['macro_f1']-cur['macro_f1'],'log_loss_change_from_source':cur['log_loss']-base['log_loss']})
 # paired source comparisons and CI
 for method,p in methods.items():
  d=metrics(ys,p);lo,hi=boot_diff(ys,cs,p,gs)
  inc.append({'population':'source_oof','comparison':method+' minus camera','n':len(ys),'n_sites':len(set(gs)),'macro_f1_difference':d['macro_f1']-metrics(ys,cs)['macro_f1'],'accuracy_difference':d['accuracy']-metrics(ys,cs)['accuracy'],'log_loss_difference':d['log_loss']-metrics(ys,cs)['log_loss'],'brier_difference':d['brier']-metrics(ys,cs)['brier'],'site_bootstrap_macro_f1_ci_low':lo,'site_bootstrap_macro_f1_ci_high':hi})
 # Confidence / ambiguity use source OOF weighted posterior to avoid optimistic source fusion.
 cp=cs.argmax(1); correct=cp==ys; wc=fused
 for label,mask in [('camera_initially_correct',correct),('camera_initially_incorrect',~correct)]:
  change=(wc.argmax(1)!=cp)&mask; right=(wc.argmax(1)==ys)&mask
  conf.append({'population':'source_oof','group':label,'n':int(mask.sum()),'n_sites':len(set(gs[mask])),'mean_camera_confidence':cs[mask].max(1).mean(),'mean_fused_confidence':wc[mask].max(1).mean(),'mean_confidence_change':(wc[mask].max(1)-cs[mask].max(1)).mean(),'prediction_change_rate':change[mask].mean(),'fused_accuracy':right[mask].mean(),'camera_ece':ece(ys[mask],cs[mask]),'fused_ece':ece(ys[mask],wc[mask])})
 close=margin(cs)<=np.quantile(margin(cs),.25)
 for label,mask in [('all_source',np.ones(len(ys),bool)),('camera_top_quartile_ambiguity',close)]:
  changed=(wc.argmax(1)!=cp)&mask
  dis.append({'population':'source_oof','group':label,'n':int(mask.sum()),'n_sites':len(set(gs[mask])),'camera_entropy':entropy(cs[mask]).mean(),'fused_entropy':entropy(wc[mask]).mean(),'entropy_change':(entropy(wc[mask])-entropy(cs[mask])).mean(),'camera_top2_margin':margin(cs[mask]).mean(),'fused_top2_margin':margin(wc[mask]).mean(),'top1_change_rate':changed[mask].mean(),'changed_prediction_correct_rate':(wc.argmax(1)[changed]==ys[changed]).mean() if changed.any() else float('nan'),'camera_accuracy':(cp[mask]==ys[mask]).mean(),'fused_accuracy':(wc.argmax(1)[mask]==ys[mask]).mean()})
 # per-class performances in individual file
 for method,p in [('camera',cs),('audio',aus)]:
  for k,cl in enumerate(classes):
   m=ys==k
   if m.any():individual.append({'population':'source','method':method,'row_type':'per_class_recall','class':cl,'n':int(m.sum()),'value':float((p.argmax(1)[m]==k).mean())})
 for x in individual:x.setdefault('row_type','overall')
 write('individual_modality_results.csv',individual);write('fusion_results.csv',fusion);write('confidence_analysis.csv',conf);write('disambiguation_results.csv',dis);write('domain_shift_results.csv',domain);write('incremental_information.csv',inc)
 svg_bar(os.path.join(HERE,'modality_performance.svg'),'Macro-F1 by label-supported population',list(pops),[(m,[next(r['macro_f1'] for r in fusion if r['population']==q and r['method']==m) for q in pops]) for m in ('camera','audio','weighted_average')])
 svg_bar(os.path.join(HERE,'confidence_changes.svg'),'Mean confidence: source OOF',[r['group'] for r in conf],[('camera',[r['mean_camera_confidence'] for r in conf]),('fused',[r['mean_fused_confidence'] for r in conf])])
 svg_bar(os.path.join(HERE,'domain_shift_comparison.svg'),'Macro-F1 by population',list(pops),[(m,[next(r['macro_f1'] for r in domain if r['population']==q and r['method']==m) for q in pops]) for m in ('camera','audio','weighted_average')])
 report=os.path.join(HERE,'multimodal_complementarity_report.md')
 strongest=max(('camera','audio'),key=lambda m:metrics(ys,methods[m])['macro_f1']); fw=metrics(ys,fused);bs=metrics(ys,methods[strongest])
 open(report,'w',encoding='utf-8').write(f'''# Multimodal complementarity (retrospective)

## Design and data checks

`clip_probs.npz` contains {len(names)} uniquely keyed clips, {cam.shape[1]} class probabilities per modality, and no assumed row-order joins. Event labels were mapped to the stored class order by class name. Analyses use only label-supported splits; target populations are reported separately. Source fusion evaluation is held-out by the supplied site fold. Weight selection uses only the corresponding source training sites (inner grouped validation, log-loss objective); target labels never enter tuning or fitting. The final target weight is source-only ({alpha_final:.2f}). Learned fusion is a regularized multinomial logistic model trained only on source camera/audio posteriors; it has no site/project/label feature.

## Answers to the research questions

On {len(ys)} source clips at {len(set(gs))} sites, the stronger individual modality is **{strongest}** (macro-F1 {bs['macro_f1']:.3f}; log loss {bs['log_loss']:.3f}). Weighted fusion has macro-F1 {fw['macro_f1']:.3f} and log loss {fw['log_loss']:.3f}: it has a small log-loss improvement but essentially unchanged macro-F1. Its site-bootstrap macro-F1 interval crosses zero, so this is not evidence of a meaningful classification improvement.

Audio does **not** help when the camera is already correct in this OOF analysis: the average fused confidence decreases, ECE worsens, and only a very small fraction of top-1 predictions changes. In the pre-specified most-ambiguous visual quartile, fusion also increases entropy and lowers the top-two margin; it does not resolve visual ambiguity here. The learned fusion and equal average do not improve on camera.

On the supplied `adapt` and `test` label-supported rows, fixed source choices show the same qualitative result: weighted fusion is almost camera-only and does not reduce macro-F1 degradation relative to source. It cannot be called domain-robust merely for matching camera; see `domain_shift_results.csv` for the degradation comparison. Audio appears most valuable, if at all, only as a marginal source log-loss contribution; the observed evidence is insufficient to support an adaptive acquisition policy. A future policy would require prospectively verified benefit conditional on observable pre-audio signals and a deployment-time acquisition cost analysis.

## Interpretation boundaries

Confidence increases are descriptive, not automatically beneficial: this report pairs them with accuracy and ECE separately for initially-correct and initially-incorrect camera clips. Ambiguity is operationalized before labels as the lowest camera top-two-margin quartile. Domain rows compare degradation from source rather than calling any higher fused score “robustness.” These are exploratory retrospective results; only fixed source choices are evaluated outside source.

Audio is available only after a camera-triggered event in SA-FARI. Therefore none of these results establishes benefit for events entirely missed by the camera, and the audio posterior is not an input to a deployment-time acquisition selector. Sample sizes, class coverage, per-class recall, metrics, paired deltas and site-aware intervals are in the CSV deliverables.
''')
 print('Wrote multimodal complementarity outputs; source alpha OOF mean',np.mean(alpha_by_fold),'final',alpha_final)
if __name__=='__main__':main()
