"""Create the human-readable report from run_nested_site_cv.py outputs."""
import csv, os, statistics
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"outputs")
def read(n): return list(csv.DictReader(open(os.path.join(OUT,n),encoding="utf-8")))
def f(x): return float(x)
def main():
 m=read("ranking_metrics.csv"); b=read("budget_results.csv"); folds=read("fold_metrics.csv")
 deterministic=[r for r in b if r["method"]!="random"]
 comparisons=[]
 for method in ("lr_camera","lr_contextual","xgb_contextual"):
  for metric in ("roc_auc","average_precision"):
   base={r["outer_fold"]:f(r[metric]) for r in folds if r["method"]=="camera_uncertainty"}
   other={r["outer_fold"]:f(r[metric]) for r in folds if r["method"]==method}
   d=[other[k]-base[k] for k in sorted(base)]
   comparisons.append(dict(comparator=method,reference="camera_uncertainty",metric=metric,n_folds=5,mean_paired_difference=statistics.mean(d),wins=sum(x>0 for x in d),fold_differences=";".join("%.5f"%x for x in d)))
 with open(os.path.join(OUT,"paired_fold_comparisons.csv"),"w",newline="",encoding="utf-8") as h:
  w=csv.DictWriter(h,fieldnames=comparisons[0]);w.writeheader();w.writerows(comparisons)
 def metric(method,variant): return next(r for r in m if r['method']==method and r['score_variant']==variant)
 with open(os.path.join(OUT,"final_report.md"),"w",encoding="utf-8") as h:
  h.write("# Member 2 — Nested site-grouped audio-usefulness experiment\n\n")
  h.write("## Setup, validation, and leakage prevention\n\nThe four selectors are E1 camera uncertainty (`1-max camera posterior`), E2 camera-only L2 logistic regression, E3 contextual L2 logistic regression, and E4 contextual XGBoost. E2 uses the class posterior, entropy, and top-two margin. E3/E4 add `is_ir`, one-hot placement, and a fixed 32-D mean-pooled visual projection. No audio-derived feature, correctness/outcome field, label, target/adapt row, site ID, or assumed row ordering is used as a feature. Events, embeddings, and posterior arrays are checked and joined by `video_name`.\n\nFive supplied site-disjoint folds are the outer evaluation. Within each outer training partition, GroupKFold over sites generates inner OOF scores used for AP-based hyperparameter selection and Platt calibration. Thus outer validation labels are never used to select, train, or calibrate. Global pooling ranks cross-fitted calibrated probabilities; fold-stratified ranking ranks the raw score inside each outer fold. The common retrospective cohort contains 1,785 audio-available source clips and 147 useful clips.\n\nE2 tunes `C` in {0.03, 0.1, 0.3, 1}. E4 is intentionally conservative: depth {1,2}, 30/60 estimators, learning rate .05, `min_child_weight=5`, `reg_lambda=10`, `reg_alpha=.2`, and 0.8 row/column subsampling. No early stopping is used, to avoid creating another label-bearing validation role with only 147 positives.\n\n## OOF ranking and calibration\n\n| Method | Score | ROC-AUC | AP | Brier | Log loss |\n|---|---|---:|---:|---:|---:|\n")
  for r in m: h.write("| %s | %s | %.4f | %.4f | %.4f | %.4f |\n"%(r['method'],r['score_variant'],f(r['roc_auc']),f(r['average_precision']),f(r['brier']),f(r['log_loss'])))
  h.write("\nRaw scores are the appropriate ranking view; calibrated scores are reported separately because calibration can change rankings across folds. E1 is strongest by raw AP (0.1557); neither contextual model improves it.\n\n")
  h.write("## Fixed-budget results\n\n")
  for strategy in ("global_pooled_calibrated","fold_stratified_raw"):
   h.write("### %s\n\n| Budget | Method | Calls | Useful | Precision | Recall | Correction rate |\n|---:|---|---:|---:|---:|---:|---:|\n"%strategy)
   for budget in (.1,.2,.3,.5):
    for method in ("camera_uncertainty","lr_camera","lr_contextual","xgb_contextual"):
     r=next(x for x in deterministic if x['strategy']==strategy and x['method']==method and f(x['budget'])==budget)
     h.write("| %d%% | %s | %s | %s | %.2f%% | %.2f%% | %.2f%% |\n"%(budget*100,method,r['audio_calls'],r['useful_captured'],100*f(r['precision']),100*f(r['recall']),100*f(r['correction_rate'])))
   h.write("\n")
  h.write("### Random reference (100 seeds)\n\n| Strategy | Budget | Useful captured, mean +/- SD | Precision, mean +/- SD |\n|---|---:|---:|---:|\n")
  for strategy in ("global_pooled_calibrated","fold_stratified_raw"):
   for budget in (.1,.2,.3,.5):
    rs=[x for x in b if x['method']=='random' and x['strategy']==strategy and f(x['budget'])==budget]
    cap=[f(x['useful_captured']) for x in rs]; pre=[f(x['precision']) for x in rs]
    h.write("| %s | %d%% | %.2f +/- %.2f | %.2f%% +/- %.2f%% |\n"%(strategy,budget*100,statistics.mean(cap),statistics.pstdev(cap),100*statistics.mean(pre),100*statistics.pstdev(pre)))
  h.write("\nRandom selections use the same eligible clips and fixed seeds in `budget_results.csv`.\n\n")
  h.write("## Paired folds and site-aware uncertainty\n\n| Comparator vs E1 | Metric | Mean paired fold difference | Folds won |\n|---|---|---:|---:|\n")
  for r in comparisons: h.write("| %s | %s | %.4f | %s/5 |\n"%(r['comparator'],r['metric'],f(r['mean_paired_difference']),r['wins']))
  h.write("\n`site_bootstrap_ci.csv` gives 1,000 site-resampled 95% CIs; `fold_metrics.csv` and `paired_fold_comparisons.csv` preserve all paired inputs. The small number of positives, expert-derived target, and finite site set make close differences inconclusive. The evidence here does not support selecting a more complex contextual model over camera uncertainty. Source-OOF calibration is not assumed to transport under domain shift.\n")
if __name__=='__main__': main()
