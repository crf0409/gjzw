# Three-Corpus Public Status Refresh

- Overall status: ok
- Steps: 9

| Step | Return code | Seconds |
|---|---:|---:|
| public_cv_status | 0 | 0.13 |
| public_cv_remaining | 0 | 0.132 |
| public_robustness_queue | 0 | 0.082 |
| public_calibration_rotation_queue | 0 | 1.97 |
| public_followup_summary | 0 | 0.191 |
| three_corpus_runbook_status | 0 | 0.131 |
| three_corpus_figures | 0 | 15.548 |
| figure_rotation_audit | 0 | 0.194 |
| three_corpus_completion_audit | 0 | 0.332 |

## Output Tails

### public_cv_status

```text
# Public-Corpus CV Parity Status

This status file tracks whether ASP_clean and AS25_clean have the same 5-fold x 3-seed confirmation depth as AL6.

| Dataset | Role | Status | Folds | Test accuracy | Macro-F1 | Summary |
|---|---|---|---:|---:|---:|---|
| ASP_clean | baseline | complete | 15/15 | 73.04 % +/- 0.47 % | 73.02 % +/- 0.40 % | `outputs/cv_asp_baseline/public_cv_parity_v1/resnet50/cv_summary.json` |
| ASP_clean | aafnet | complete | 15/15 | 73.59 % +/- 0.52 % | 73.63 % +/- 0.52 % | `outputs/cv_asp_aafnet/public_cv_parity_v1/resnet50/cv_summary.json` |
| AS25_clean | baseline | complete | 15/15 | 66.82 % +/- 0.68 % | 66.63 % +/- 0.71 % | `outputs/cv_as25_baseline/public_cv_parity_v1/resnet50/cv_summary.json` |
| AS25_clean | aafnet | complete | 15/15 | 67.66 % +/- 0.60 % | 67.47 % +/- 0.64 % | `outputs/cv_as25_aafnet/public_cv_parity_v1/resnet50/cv_summary.json` |

Overall status: **complete**.


## Active run monitor

| Run | PID | Alive | Elapsed | Log |
|---|---:|---|---:|---|
| public_cv_parity_v1 | 3815493 | False | n/a | `outputs/public_cv_parity/logs/public_cv_parity_v1_20260619_121840.log` |
| watch_public_cv_progress | 3970840 | False | n/a | `n/a` |

# Public-Corpus CV Smoke Status

Smoke runs use 2 folds x 1 seed x 1 epoch and are only used to verify the training path.

| Dataset | Role | Status | Folds | Test accuracy | Summary |
|---|---|---|---:|---:|---|
| ASP_clean | baseline | complete | 2/2 | 55.02 % +/- 1.40 % | `outputs/cv_smoke_asp_baseline/20260619_114248/resnet50/cv_summary.json` |
| ASP_clean | aafnet | complete | 2/2 | 49.99 % +/- 0.91 % | `outputs/cv_smoke_asp_aafnet/20260619_114434/resnet50/cv_summary.json` |
| AS25_clean | baseline | complete | 2/2 | 34.07 % +/- 3.17 % | `outputs/cv_smoke_as25_baseline/20260619_114622/resnet50/cv_summary.json` |
| AS25_clean | aafnet | complete | 2/2 | 30.80 % +/- 2.40 % | `outputs/cv_smoke_as25_aafnet/20260619_114745/resnet50/cv_summary.json` |

Smoke overall status: **complete**.
```

### public_cv_remaining

```text
# Public-Corpus CV Remaining Queue

This file enumerates the exact 5-fold x 3-seed cells still required before ASP_clean and AS25_clean can be treated as AL6-level statistical peers.

| Dataset | Role | Complete | Missing | Next missing cells |
|---|---|---:|---:|---|
| ASP_clean | baseline | 15/15 | 0 | none |
| ASP_clean | aafnet | 15/15 | 0 | none |
| AS25_clean | baseline | 15/15 | 0 | none |
| AS25_clean | aafnet | 15/15 | 0 | none |

## Active Runs

| Run | PID | Alive | Elapsed | Log |
|---|---:|---|---:|---|
| public_cv_parity_v1 | 3815493 | False | n/a | `outputs/public_cv_parity/logs/public_cv_parity_v1_20260619_121840.log` |
| watch_public_cv_progress | 3970840 | False | n/a | `n/a` |
```

### public_robustness_queue

```text
# Public-Corpus Robustness Attribution Queue

This file tracks fold-level robustness attribution readiness for ASP_clean and AS25_clean.

- CV run id: `public_cv_parity_v1`
- Source data: `paper/figures/nature_source_data/public_robustness_attribution_queue_source.csv`

| Dataset | Role | CV checkpoints ready | Robustness complete | Waiting for CV |
|---|---|---:|---:|---:|
| ASP_clean | baseline | 15/15 | 15/15 | 0/15 |
| ASP_clean | aafnet | 15/15 | 15/15 | 0/15 |
| AS25_clean | baseline | 15/15 | 15/15 | 0/15 |
| AS25_clean | aafnet | 15/15 | 15/15 | 0/15 |
```

### public_calibration_rotation_queue

```text
# Public-Corpus Calibration/Rotation Queue

This file tracks fold-level calibration and 24-angle rotation readiness for ASP_clean and AS25_clean.

- CV run id: `public_cv_parity_v1`
- Source data: `paper/figures/nature_source_data/public_calibration_rotation_queue_source.csv`

| Dataset | Role | CV checkpoints ready | Calibration complete | Rotation complete | Waiting for CV |
|---|---|---:|---:|---:|---:|
| ASP_clean | baseline | 15/15 | 15/15 | 15/15 | 0/15 |
| ASP_clean | aafnet | 15/15 | 15/15 | 15/15 | 0/15 |
| AS25_clean | baseline | 15/15 | 15/15 | 15/15 | 0/15 |
| AS25_clean | aafnet | 15/15 | 15/15 | 15/15 | 0/15 |
```

### public_followup_summary

```text
# Public Follow-Up Probe Summary

- CV run id: `public_cv_parity_v1`
- This summary only aggregates completed fold-level probe JSON files.

| Dataset | Role | Robustness | Calibration | Rotation |
|---|---|---:|---:|---:|
| ASP_clean | baseline | 15/15 | 15/15 | 15/15 |
| ASP_clean | aafnet | 15/15 | 15/15 | 15/15 |
| AS25_clean | baseline | 15/15 | 15/15 | 15/15 |
| AS25_clean | aafnet | 15/15 | 15/15 | 15/15 |

## Source Data

- `public_followup_probe_status_source.csv`
- `public_robustness_fold_metrics_source.csv`
- `public_calibration_fold_metrics_source.csv`
- `public_rotation_fold_metrics_source.csv`
- `public_robustness_summary_source.csv`
- `public_calibration_summary_source.csv`
- `public_rotation_summary_source.csv`
```

### three_corpus_runbook_status

```text
wrote: outputs/three_corpus_completion/missing_experiment_runbook.md
wrote: paper/figures/nature_source_data/three_corpus_live_status_source.csv
```

### three_corpus_figures

```text
[completing three-corpus figures]
  -> /home/siton02/disk_sdg/md0_backup_2026-04-28/crf/gjzw/paper/figures/F_TC0_three_corpus_samples.png
  -> /home/siton02/disk_sdg/md0_backup_2026-04-28/crf/gjzw/paper/figures/F_TC1_three_corpus_audit.png
  -> /home/siton02/disk_sdg/md0_backup_2026-04-28/crf/gjzw/paper/figures/F_TC2_three_corpus_performance.png
  -> /home/siton02/disk_sdg/md0_backup_2026-04-28/crf/gjzw/paper/figures/F_TC3_three_corpus_confusions.png
  -> /home/siton02/disk_sdg/md0_backup_2026-04-28/crf/gjzw/paper/figures/F_TC4_three_corpus_training.png
  -> /home/siton02/disk_sdg/md0_backup_2026-04-28/crf/gjzw/paper/figures/F_TC5_three_corpus_probe_matrix.png
  -> /home/siton02/disk_sdg/md0_backup_2026-04-28/crf/gjzw/paper/figures/F_TC8_public_experiment_coverage.png
[ok] QA written to /home/siton02/disk_sdg/md0_backup_2026-04-28/crf/gjzw/paper/figures/three_corpus_completion_qa.md
```

### figure_rotation_audit

```text
rotation rows: 46
errors: 0
wrote: paper/figures/figure_rotation_audit.md
wrote: paper/figures/nature_source_data/figure_rotation_audit.csv
```

### three_corpus_completion_audit

```text
wrote: outputs/three_corpus_completion/three_corpus_parity_completion_audit.md
complete: True
```
