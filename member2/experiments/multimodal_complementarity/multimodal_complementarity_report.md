# Multimodal complementarity (retrospective)

## Design and data checks

`clip_probs.npz` contains 2311 uniquely keyed clips, 23 class probabilities per modality, and no assumed row-order joins. Event labels were mapped to the stored class order by class name. Analyses use only label-supported splits; target populations are reported separately. Source fusion evaluation is held-out by the supplied site fold. Weight selection uses only the corresponding source training sites (inner grouped validation, log-loss objective); target labels never enter tuning or fitting. The final target weight is source-only (0.95). Learned fusion is a regularized multinomial logistic model trained only on source camera/audio posteriors; it has no site/project/label feature.

## Answers to the research questions

On 1786 source clips at 184 sites, the stronger individual modality is **camera** (macro-F1 0.507; log loss 1.575). Weighted fusion has macro-F1 0.507 and log loss 1.554: it has a small log-loss improvement but essentially unchanged macro-F1. Its site-bootstrap macro-F1 interval crosses zero, so this is not evidence of a meaningful classification improvement.

Audio does **not** help when the camera is already correct in this OOF analysis: the average fused confidence decreases, ECE worsens, and only a very small fraction of top-1 predictions changes. In the pre-specified most-ambiguous visual quartile, fusion also increases entropy and lowers the top-two margin; it does not resolve visual ambiguity here. The learned fusion and equal average do not improve on camera.

On the supplied `adapt` and `test` label-supported rows, fixed source choices show the same qualitative result: weighted fusion is almost camera-only and does not reduce macro-F1 degradation relative to source. It cannot be called domain-robust merely for matching camera; see `domain_shift_results.csv` for the degradation comparison. Audio appears most valuable, if at all, only as a marginal source log-loss contribution; the observed evidence is insufficient to support an adaptive acquisition policy. A future policy would require prospectively verified benefit conditional on observable pre-audio signals and a deployment-time acquisition cost analysis.

## Interpretation boundaries

Confidence increases are descriptive, not automatically beneficial: this report pairs them with accuracy and ECE separately for initially-correct and initially-incorrect camera clips. Ambiguity is operationalized before labels as the lowest camera top-two-margin quartile. Domain rows compare degradation from source rather than calling any higher fused score “robustness.” These are exploratory retrospective results; only fixed source choices are evaluated outside source.

Audio is available only after a camera-triggered event in SA-FARI. Therefore none of these results establishes benefit for events entirely missed by the camera, and the audio posterior is not an input to a deployment-time acquisition selector. Sample sizes, class coverage, per-class recall, metrics, paired deltas and site-aware intervals are in the CSV deliverables.
