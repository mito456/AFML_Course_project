# Final project results index

## Authoritative reports

The project has two separate experiments:

| Track | Authoritative report | Authoritative outputs | Supported conclusion |
|---|---|---|---|
| SA-FARI / Member 2 | `SAFARI_MEMBER2_FINAL_REPORT.md` | `member2/experiments/final_validation/results/policy_comparison.csv`, `paired_site_bootstrap.csv`, `target_transfer.csv`, `source_selected_transfer_policy.csv` | Continuous incremental value has the higher 30% point estimate than camera uncertainty, but its paired site-bootstrap benefit interval crosses zero; superiority is not established. Frozen target transfer is mixed and not convincing generalization. |
| Taiwan / Member 1 | `TAIWAN_MEMBER1_FINAL_REPORT.md` | `member1_audit_source/outputs/taiwan/station/coverage_overall.csv`, `transfer_results.csv`, `equal_budget_ci.csv`; code in `member1_audit_source/taiwan/` | Recorder-only detections dominate in this deployment. Equal-budget adaptive-vs-fixed intervals all cross zero; adaptive advantage is not established and it uses selected target outcomes online. |

See `SHARED_PROJECT_PROTOCOL_COMPARISON.md` for the deliberately non-numeric comparison of the two protocols.

## Provenance audit

| File / artifact | Track | Experiment/run | Contents | Status |
|---|---|---|---|---|
| `member2/experiments/final_validation/results/policy_comparison.csv` | SA-FARI | Final validation, source nested OOF | All policies and 10/20/30/50% budgets | Authoritative |
| `member2/experiments/final_validation/results/paired_site_bootstrap.csv` | SA-FARI | Final validation | Paired site bootstrap, seed 20261115, 2,000 replicates | Authoritative |
| `member2/experiments/final_validation/results/target_transfer.csv` | SA-FARI | Final validation | Frozen source-selected target evaluation | Authoritative |
| `member2/experiments/final_validation/results/source_selected_transfer_policy.csv` | SA-FARI | Final validation | Selected frozen policy configuration | Authoritative |
| `member2/member2_final_validation_report.md` | SA-FARI | Earlier reconciliation report | Narrative/table with values that conflict with final CSV | Superseded; do not cite numerical table |
| `member2/experiments/final_validation/final_report.md` | SA-FARI | Earlier final-validation report | Earlier 30% table and 1,000-bootstrap wording | Superseded; do not cite numerical table |
| `member2/experiments/incremental_audio_value/`, `shim_inspired_rejector/`, `outputs/`, `conditional_policy/` | SA-FARI | Earlier/checkpoint experiments | Exploratory policies and reports | Supporting / superseded for final claims |
| `member2/member2_protocol_alignment_audit.md` | SA-FARI | Audit | Alignment/provenance limitations | Supporting |
| `member1_audit_source/outputs/taiwan/station/coverage_overall.csv` | Taiwan | Station results | Sensor coverage and station bootstrap | Authoritative |
| `member1_audit_source/outputs/taiwan/station/transfer_results.csv` | Taiwan | Station transfer | Fixed, adaptive, random expectation, oracle | Authoritative |
| `member1_audit_source/outputs/taiwan/station/equal_budget_ci.csv` | Taiwan | Equal-budget comparison | Station-bootstrap fixed/adaptive differences | Authoritative; present 30/40/60% rows |
| `member1_audit_source/outputs/taiwan/station/station_results.md` | Taiwan | Station results narrative | Methods and final presentation interpretation | Supporting authoritative narrative |
| `m1_station_results_report.md`, `station_results.md`, `m1_station_results.zip` | Taiwan | Packaged/summary result | Root-level summary/package | Supporting; provenance of package timestamp/version is not independently established |
| `member2/member1_implementation_audit.md`, `member2/member1_to_member2_protocol_mapping.md`, `member2/member1_open_questions.md` | Shared/Taiwan audit | Code audit | Implementation evidence and unresolved assumptions | Supporting |
| `member2/shared_protocol_comparison.md` | Shared | Earlier shared write-up | Protocol comparison | Superseded by `SHARED_PROJECT_PROTOCOL_COMPARISON.md` |

Authority was determined from result-directory provenance, code/audit linkage, content, and timestamps—not filenames. SA-FARI authoritative outputs were last modified 2026-11-15 10:46--11:02 UTC; Taiwan audited station outputs were last modified 2026-11-15 10:23 UTC. The repository revision inspected for the current checkout was `acccce7c7b86bd90f45aac5196908e2135eb1e75`; the separate Taiwan audit identifies Member 1 revision `1b11252`.

## Reconciliation record

The prior SA-FARI report table was replaced in the authoritative report with direct CSV values. In particular, continuous-incremental total benefit at 30% is **36.886**, not the earlier report value **38.226**. The corrected 30% paired-site interval for continuous incremental minus camera uncertainty, on **total incremental benefit**, is **[-5.166, 18.146]**, not an earlier summary value. It has 2,000 valid samples with seed 20261115.

Unresolved provenance limitations: SA-FARI's visual embedding has manifest-level alignment but lacks independently reconstructible upstream generation; Taiwan has no established private/newer result set, retains camera-tip-off timing/reliability and target-feedback operational assumptions, and has a raw 20% equal-budget artifact that is intentionally excluded from final presentation. No model, feature, split, selection logic, implementation, or experiment was changed or run for this audit.
