# Three-Corpus Completion Summary

The manuscript originally had the richest experimental presentation for AL6. This package completes a parallel, auditable figure layer for AL6, ASP_clean and AS25_clean using existing cached results.

## Generated figures

- `F_TC0_three_corpus_samples.png`
- `F_TC1_three_corpus_audit.png`
- `F_TC2_three_corpus_performance.png`
- `F_TC3_three_corpus_confusions.png`
- `F_TC4_three_corpus_training.png`
- `F_TC5_three_corpus_probe_matrix.png`
- `F_TC6_three_corpus_parity.png`
- `F_TC7_public_interpretability.png`
- `F_TC8_public_experiment_coverage.png`

## Evidence boundaries

- No new model training is performed by this script.
- Public-corpus performance uses completed or partial public-CV summaries where available; partial rows are labeled by completed fold count, and missing public-corpus CV cells fall back to existing 3-seed single-split runs.
- AL6 accuracy uses the existing 5-fold x 3-seed CV summary; AL6 confusion and training curves use the headline single checkpoint.
- Robustness, calibration and rotation parity for ASP_clean/AS25_clean uses existing seed-42 checkpoints only.
- Full public-corpus robustness attribution now has a resumable fold-level queue and result aggregator, but robustness JSON outputs are still pending.
- Full public-corpus calibration and rotation now have a resumable fold-level queue and result aggregator, but fold-level JSON outputs are still pending.
- Public-corpus interpretability parity uses selected ASP_clean/AS25_clean seed-42 inference traces, with selected-case and full prediction CSVs.
- Display-only rotations for inference thumbnails and GradCAM overlays are audited in `paper/figures/figure_rotation_audit.md`.
- OOD and few-shot probes are not defined for AL6 in the current downstream battery; those cells are explicitly marked n/a.

## Training-artifact support audit

| Requirement | Status | Evidence | Boundary |
|---|---|---|---|
| Three-corpus data audit and representative samples | supported | outputs/data_audit/{AL6,ASP,AS25}/audit.json; data/processed/*_mapping.csv | Source AL6 images include some orientation variation; paper sample panel uses fixed upright display examples. |
| Three-corpus clean performance comparison | partially supported | outputs/sig_collect_v2/significance.json; outputs/p1_asp_as25_summary.json; outputs/public_cv_parity/status.json | Public corpora use completed or partial public-CV summaries where available, with partial rows explicitly labeled by completed fold count; otherwise they remain 3-seed single-split fallbacks. |
| Three-corpus confusion matrices | supported | outputs/ddp_{baseline,aafnet_v2}/*/test_metrics.json; outputs/asp_as25_*/*/test_metrics.json | AL6 confusion uses the headline single checkpoint; ASP/AS25 use seed-42 reference checkpoints. |
| Three-corpus training dynamics | supported | training_curve.csv files for AL6 headline checkpoints and ASP/AS25 seed-42 runs | Reference curves only, not fold-averaged curves. |
| Three-corpus strict-test stability | supported | outputs/p2_strict_extended.json | Single seed=42 checkpoint evaluation. |
| Three-corpus downstream feature probes | partially supported | outputs/downstream/20260509_005735/results.json | Retrieval and near-duplicate probes cover all three corpora; OOD/few-shot are defined only for ASP_clean/AS25_clean as external corpora. |
| ASP/AS25 seed-42 robustness, calibration and rotation parity | partially supported | outputs/three_corpus_parity/latest/results.json; paper/figures/F_TC6_three_corpus_parity.png; outputs/public_calibration_rotation/queue.md; outputs/public_followup_probes/summary.md | Existing checkpoints support paired seed-42 probes; fold-level calibration/rotation is now queued and has a result aggregator for completed public-CV checkpoints, but fold-level probe JSON outputs are not yet complete. |
| ASP/AS25 seed-42 inference and GradCAM parity | partially supported | paper/figures/F_TC7_public_interpretability.png; paper/figures/nature_source_data/F_TC7_public_interpretability_predictions_source.csv; paper/figures/figure_rotation_audit.md | Existing checkpoints support qualitative public-corpus inference traces. Display-only rotations are audited separately; this is not a multi-seed interpretability study. |
| Full AL6-level public-corpus validation | in progress | outputs/public_cv_parity/status.md; outputs/public_cv_parity/remaining_queue.md; outputs/public_cv_parity/smoke_status.md; outputs/cv_asp_baseline/public_cv_parity_v1/resnet50/seed42_fold*_train/*/resnet50/test_metrics.json | Public CV launcher/status scanner, remaining-cell queue, resumable fold skipping, 2-fold smoke checks, and the completed ASP_clean baseline full-CV summary now exist; the remaining public-corpus baseline/AAFNet summaries are still required before claiming identical statistical depth across all three corpora. |
| ASP/AS25 full robustness attribution and multi-seed interpretability parity | in progress | scripts/run_public_robustness_attribution.py; outputs/public_robustness_attribution/queue.md; paper/figures/nature_source_data/public_robustness_attribution_queue_source.csv | Fold-level public-corpus robustness attribution is now queued and resumable for completed CV checkpoints; matched multi-seed GradCAM/activation panels remain to be generated. |
