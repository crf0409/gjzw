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
