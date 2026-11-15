"""Run the frozen final Member 2 validation from the repository root."""
from nested_validation import run_nested, load, features, model, save
from policy_comparison import compare, _metrics
from pathlib import Path
import csv
import numpy as np

HERE=Path(__file__).resolve().parent
def target_transfer():
    """Frozen continuous-delta ridge, selected by source nested validation."""
    events,src,cam,aud,vis,classes,fold,groups,y,cp,ap,delta,useful=load(); fs,_=features(src,cam,vis)
    final=model('regression','ridge',20261115); final.fit(fs['posterior_uncertainty'],delta)
    lookup={c:i for i,c in enumerate(classes)}; out=[]
    for split in ('adapt','test'):
        rr=[r for r in events if r['split']==split and r['has_audio']=='1']
        if not rr: continue
        ff,_=features(rr,cam,vis); score=final.predict(ff['posterior_uncertainty']); order=np.argsort(-score,kind='stable'); sel=np.zeros(len(rr),bool); sel[order[:int(np.ceil(.30*len(rr)))]]=True
        yy=np.array([lookup[r['species']] for r in rr]); cc=np.vstack([cam[r['video_name']] for r in rr]); aa=np.vstack([aud[r['video_name']] for r in rr]); dd=np.log(np.clip((.95*cc+.05*aa)[np.arange(len(yy)),yy],1e-12,1))-np.log(np.clip(cc[np.arange(len(yy)),yy],1e-12,1))
        out.append({'split':split,'policy':'frozen_continuous_incremental_ridge','budget':.30,'n_clips':len(rr),**_metrics(yy,cc,.95*cc+.05*aa,sel,dd)})
        out.append({'split':split,'policy':'camera_only','budget':.30,'n_clips':len(rr),**_metrics(yy,cc,.95*cc+.05*aa,np.zeros(len(rr),bool),dd)})
    save('target_transfer.csv',out); return out
def main():
    r=run_nested(); rows,cam,aud,classes,fold,groups,y,cp,ap,delta,useful,scores,unc,selections=r
    policy,ci=compare(rows,classes,fold,groups,y,cp,ap,delta,scores,unc); target=target_transfer()
    p30=[x for x in policy if x['budget']==.3]
    by={x['policy']:x for x in p30}
    lines=['# Final Member 2 validation','', '## Audit and provenance','', f'Validated {len(rows)} source audio-eligible clips from {len(set(groups))} sites. Camera and audio posteriors were joined by `video_name`; every source row declares `prob_source=oof_site_cv`, and supplied outer folds have zero site overlap. Source audio posterior is used only to construct/evaluate the retrospective 95/5 fusion target. Selectors receive camera posterior/uncertainty, visual embedding, IR and placement only.','', 'Earlier incremental and Shim experiments choose candidates using pooled source OOF results, then report that same pooled set. This is not label leakage into a held-out clip, but it can create model-selection optimism. This run corrects it visibly: each outer-fold candidate/target choice is made only from inner grouped OOF predictions on outer-training sites.','', '## Nested design','', 'Five fixed site-disjoint outer folds are shared by every method. In each fold, feature-set/model/meaningful-benefit target choices are selected by 30% inner grouped-OOF yield. The frozen model is then fit on all outer-training sites and scored once on held-out sites. No outer label enters preprocessing, selection, calibration, thresholding, or cost choice.','', '## 30% source nested-OOF policy comparison','', '| Policy | Macro-F1 | Log loss | Calls | Total benefit | Benefit/call |','|---|---:|---:|---:|---:|---:|']
    for name in ('camera_only','audio_always','random','camera_uncertainty','binary_usefulness','shim_cost_sensitive','continuous_incremental','binary_incremental'):
      x=by[name]; lines.append(f"| {name} | {x['macro_f1']:.4f} | {x['log_loss']:.4f} | {x['audio_calls']} | {x['total_incremental_benefit']:.3f} | {x['mean_incremental_benefit_per_call']:.4f} |")
    lines += ['', 'Paired site-bootstrap intervals (1,000 resamples) are in `results/paired_site_bootstrap.csv`; they resample sites with all their clips and condition on the fixed nested OOF predictions. Thus they do not fully capture split/model-selection uncertainty and are not significance claims.', '', '## Frozen target evaluation','', '`results/target_transfer.csv` applies the frozen source-selected continuous ridge separately to adaptation and test at 30%. Target labels are used only for these reported metrics, never fitting or selection. The adaptation cohort is especially small.', '', '## Decision','', 'Continuous incremental value has the strongest 30% point estimate, but the paired intervals must decide whether it reliably clears camera uncertainty. If its interval includes practically relevant harm or improvement is limited to log loss, retain camera uncertainty as the simpler reference; do not infer domain generalization from the small target sets.', '', '## Limitations','', 'Only a limited number of sites and positive/useful outcomes are available; audio experts are weak/noisy; fixed fusion is an evaluation assumption; visual/context inputs may retain site-correlated structure; and bootstrap conditional on OOF scores understates full training uncertainty.']
    (HERE/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (HERE/'final_methodology.md').write_text('# Frozen methodology\n\nPrimary target is `log p(0.95 camera + 0.05 audio at true class) - log p(camera at true class)`. 90/10 is sensitivity-only and is not selected. The policy uses only pre-audio information. Nested five-fold site-grouped validation selects candidates within each outer training split; outputs are outer OOF only. Costs `.001, .005, .01` are predefined.\n',encoding='utf8')
    (HERE/'member2_handoff.md').write_text('# Member 2 handoff\n\nResearch question: can pre-audio camera/visual/context information predict the value of acquiring audio? Formulations retained for comparison are top-1 usefulness, supervised cost-sensitive usefulness (Shim-inspired—not RL), and continuous/binary incremental log-benefit. Member 4 should consume only the selected frozen policy and its pre-audio feature transform; never audio features. Proven: keyed source OOF/site separation was checked. Suggestive: policy point estimates. Unresolved: reliable improvement over uncertainty and target transfer. See `final_report.md` and result CSVs.\n',encoding='utf8')
    (HERE/'README.md').write_text('# Final validation\n\nRun `..\\..\\..\\.venv-member2\\Scripts\\python.exe final_validation.py` from this directory (or use its absolute path from repository root). All generated outputs remain in `results/`.\n',encoding='utf8')
    print('completed',HERE/'results')
if __name__=='__main__': main()
