"""Run the frozen final Member 2 validation from the repository root."""
from nested_validation import run_nested, load, features, model, save, select_candidate
from policy_comparison import compare, _metrics
from pathlib import Path
import csv
import numpy as np

HERE=Path(__file__).resolve().parent
def target_transfer():
    """Fit a source-selected continuous policy once, then freeze it for target."""
    events,src,cam,aud,vis,classes,fold,groups,y,cp,ap,delta,useful=load(); fs,_=features(src,cam,vis)
    candidates=[(name,'ridge','delta') for name in fs]
    train=np.arange(len(src))
    choice,_=select_candidate('regression',candidates,fs,delta,delta,groups,train,np.array([r['video_name'] for r in src]),20261115)
    feature_set,kind,tag=choice
    final=model('regression',kind,20261115); final.fit(fs[feature_set],delta)
    save('source_selected_transfer_policy.csv',[{'policy':'continuous_incremental','feature_set':feature_set,'model':kind,'target':tag,'selection_data':'source_only_inner_grouped_oof','selection_budget':.30,'fit_data':'all_source_audio_eligible'}])
    lookup={c:i for i,c in enumerate(classes)}; out=[]
    for split in ('adapt','test'):
        rr=[r for r in events if r['split']==split and r['has_audio']=='1']
        if not rr: continue
        ff,_=features(rr,cam,vis); score=final.predict(ff[feature_set]); order=np.argsort(-score,kind='stable'); sel=np.zeros(len(rr),bool); sel[order[:int(np.ceil(.30*len(rr)))]]=True
        yy=np.array([lookup[r['species']] for r in rr]); cc=np.vstack([cam[r['video_name']] for r in rr]); aa=np.vstack([aud[r['video_name']] for r in rr]); dd=np.log(np.clip((.95*cc+.05*aa)[np.arange(len(yy)),yy],1e-12,1))-np.log(np.clip(cc[np.arange(len(yy)),yy],1e-12,1))
        out.append({'split':split,'policy':'frozen_source_selected_continuous_incremental','feature_set':feature_set,'budget':.30,'n_clips':len(rr),**_metrics(yy,cc,.95*cc+.05*aa,sel,dd)})
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
    root_report=HERE.parents[1]/'member2_final_validation_report.md'
    table=['| Policy | 10% benefit | 20% benefit | 30% benefit | 50% benefit |','|---|---:|---:|---:|---:|']
    for name in ('camera_only','audio_always','random','camera_uncertainty','binary_usefulness','continuous_incremental','binary_incremental','oracle'):
        vals=[next(x for x in policy if x['policy']==name and x['budget']==b)['total_incremental_benefit'] for b in (.1,.2,.3,.5)]
        table.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    target_table=['| Split | Clips | Rate | Macro-F1 | Log loss | Incremental benefit |','|---|---:|---:|---:|---:|---:|']
    for x in target:
        if x['policy']=='camera_only': continue
        target_table.append(f"| {x['split']} | {x['n_clips']} | {x['audio_acquisition_rate']:.3f} | {x['macro_f1']:.4f} | {x['log_loss']:.4f} | {x['total_incremental_benefit']:.3f} |")
    root_report.write_text('''# Member 2 final validation report

## Experimental protocol

The decision unit is a camera-triggered clip with audio available. Policies process audio for an exact per-outer-fold `ceil(budget × eligible clips)` at 10%, 20%, 30%, or 50%; the common source cohort has {N_CLIPS} clips from {N_SITES} sites. Source selector fitting/configuration uses source data only with five supplied site-disjoint outer folds and grouped inner folds. Target adaptation and test are frozen evaluations: their labels are read only to calculate metrics.

## Validation audit

Camera/audio posteriors are key-joined by `video_name`; source posterior provenance is `oof_site_cv`, and supplied outer folds have no site overlap. Selector inputs are camera posterior/uncertainty, visual embedding, IR, and placement; no audio-derived input, label, site/project ID, or target metadata enters a selector. `visual_emb.npy` is mapped through the `feat_index.csv` manifest after verifying cardinality, uniqueness, key-set equality, and identical manifest/event order. This establishes the repository's declared alignment, but the checkout contains no embedding-generation script or row identifiers inside the NPY; independent reconstruction from upstream generation remains unavailable.

## Equal-budget policy comparison

Incremental benefit is the change in true-class log probability from fixed 95/5 camera/audio fusion versus camera-only. All policies use the same per-fold exact-budget convention; consequently even Oracle can include a negative-benefit clip when fewer than the budget count have positive benefit. Oracle ranks realized retrospective benefit and is an upper bound, not deployable.

{POLICY_TABLE}

Camera uncertainty is the simple reference. Binary usefulness predicts `audio correct AND camera wrong`; continuous incremental value predicts fusion-objective improvement. `shim_cost_sensitive` is operationally identical to binary usefulness in this run (same target, candidates, selection criterion and ranked decisions), so it is not an independent method. Random is empirical fixed-seed random acquisition, always audio is maximum acquisition, and camera-only is zero acquisition.

## Statistical comparison

`results/paired_site_bootstrap.csv` contains paired whole-site bootstrap comparisons at the prespecified 30% operating point: continuous vs uncertainty; binary incremental vs uncertainty; continuous vs binary incremental; and each learned policy vs random and camera-only. It uses 2,000 resamples, seed 20261115, identical resampled sites within each pair, 95% percentile intervals, and records valid sample counts. These intervals condition on fixed nested OOF predictions; they do not include uncertainty from repeated fitting, split selection, or policy/model selection. Overlapping intervals are not evidence of equivalence. Point estimates for all four budgets are retained above.

## Target transfer

`results/source_selected_transfer_policy.csv` records a continuous configuration selected with source-only inner grouped OOF data and fit once on all source clips. `results/target_transfer.csv` applies that frozen policy separately to adaptation and test at 30%; it reports clip count, acquisition rate, macro-F1, log loss, and incremental benefit. It is a source-selected frozen-policy evaluation, not target adaptation: no target feedback, thresholding, refitting, or selection occurred.

{TARGET_TABLE}

## Limitations and interpretation

Audio experts are imperfect and results depend on fixed late fusion. Domain shift and small target adaptation data limit transfer claims. Camera-triggered sampling cannot establish audio's value for events the camera never triggers. The embedding alignment has manifest-level but not independently reconstructible provenance. Bootstrap intervals understate full pipeline uncertainty, and recording versus processing budgets are different cost proxies.

Continuous incremental value has the best 30% source point estimate but earlier paired intervals included harm versus camera uncertainty, so it is promising rather than established. Oracle headroom measures retrospective opportunity under this fusion outcome, not attainable deployment performance. Target transfer is descriptive and not convincing evidence of generalization. The defensible claim is that frozen, pre-audio clip-level ranking can prioritize retrospective fusion benefit on this source protocol; it does not establish broadly adaptive audio acquisition.
'''.replace('{N_CLIPS}',str(len(rows))).replace('{N_SITES}',str(len(set(groups)))).replace('{POLICY_TABLE}','\n'.join(table)).replace('{TARGET_TABLE}','\n'.join(target_table)),encoding='utf8')
    (HERE.parents[1]/'member2_protocol_alignment_audit.md').write_text('''# Member 2 protocol alignment audit

| Component | Current implementation | Issue identified | Correction required |
|---|---|---|---|
| Nested validation | Five supplied grouped outer folds; grouped inner OOF selection | Confirmed site-disjoint source folds | Retained; outer OOF used for results |
| Posterior provenance | `video_name` join; source `oof_site_cv` | None found | Retained keyed validation |
| Visual embeddings | NPY rows were assigned from event position | Positional dependency was implicit | Now map via verified `feat_index` manifest; upstream generation unavailable |
| Policy comparison | Exact per-fold budgets; paired site bootstrap | No oracle; 1,000 resamples | Added oracle and 2,000 paired site resamples |
| Target transfer | Hardcoded posterior-uncertainty ridge | Did not reproduce source-selected configuration | Source-only grouped-OOF configuration is now selected, fit once, and frozen |
| Method naming | Binary usefulness and Shim had identical scored implementation | Presented as distinct names | Retained code names, marked operationally equivalent |

Confirmed facts: source rows are OOF site-CV; source sites do not cross supplied folds; posterior keys are unique; selector inputs exclude audio-derived fields, labels, site/project IDs, and target rows. Assumption explicitly retained: fixed 95/5 late fusion defines retrospective benefit. `feat_index` gives an explicit row-to-clip manifest and matches `clip_events` order/key set, but no upstream embedding-generation code or embedded identifiers are present, so independent provenance beyond this manifest cannot be verified. Visual-feature conclusions should retain that limitation.
''',encoding='utf8')
    (HERE.parents[1]/'shared_protocol_comparison.md').write_text('''# Shared protocol comparison

| Protocol dimension | Member 1: Station-level | Member 2: Clip-level |
|---|---|---|
| Decision unit | Station-hour | Camera-triggered clip |
| Acquisition decision | Whether recorder is active | Whether audio is processed/used |
| Always-available modality | Camera | Camera |
| Additional modality | Acoustic recorder | Audio |
| Primary evaluation metric | Detection capture rate | Incremental log loss / classification performance |
| Acquisition cost proxy | Recording budget (%% recorder-on) | Audio processing rate (%% clips processed) |
| Random baseline | Analytical expected capture or implemented random schedule (report-specific) | Empirical random selection |
| Source-only baseline | Fixed source-derived schedule | Camera uncertainty / source-trained selector |
| Adaptive method | Target-updating adaptive schedule (selected-outcome feedback assumed) | Frozen source-trained selector |
| Oracle | Retrospective target-label oracle | Retrospective clip-level oracle |
| Target feedback | Selected recorder outcomes update future decisions | No target-label feedback |
| Bootstrap unit | Station | Site |

These are protocol alignments, not equivalent algorithms. Member 1 makes station-hour recording decisions; Member 2 makes clip-level audio-processing decisions. Their cost proxies and primary benefit metrics differ. Member 1's adaptive policy uses selected target recorder outcomes for online updating, whereas Member 2 target evaluation remains frozen. Member 2 covers only camera-triggered clips and cannot estimate audio value for events never detected by the camera.
''',encoding='utf8')
    (HERE/'final_report.md').write_text(root_report.read_text(encoding='utf8'),encoding='utf8')
    (HERE/'README.md').write_text('# Final validation\n\nRun `..\\..\\..\\.venv-member2\\Scripts\\python.exe final_validation.py` from this directory (or use its absolute path from repository root). All generated outputs remain in `results/`.\n',encoding='utf8')
    print('completed',HERE/'results')
if __name__=='__main__': main()
