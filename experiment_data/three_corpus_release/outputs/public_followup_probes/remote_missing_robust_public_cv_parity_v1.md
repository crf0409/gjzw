# Public Follow-Up Probe Pool

- Label: `remote_missing_robust_public_cv_parity_v1`
- CV all complete: True
- Devices: `cuda:0, cuda:1, cuda:2, cuda:3`
- Jobs: 4
- Failed jobs: 0

## Setup Steps

| Step | Return code | Seconds |
|---|---:|---:|
| public_robustness_queue | 0 | 0.093 |
| public_calibration_rotation_queue | 0 | 2.101 |
| public_followup_summary | 0 | 0.117 |
| public_robustness_queue | 0 | 0.092 |
| public_calibration_rotation_queue | 0 | 2.152 |

## Jobs

| Stage | Cell | Device | Return code | Seconds | Log |
|---|---|---|---:|---:|---|
| robustness | AS25_clean aafnet s2024-f1 | cuda:0 | 0 | 110.672 | `outputs/public_followup_probes/logs/remote_missing_robust_public_cv_parity_v1_worker0_robust_as25_aafnet_s2024_f1.log` |
| robustness | AS25_clean aafnet s2024-f2 | cuda:1 | 0 | 116.371 | `outputs/public_followup_probes/logs/remote_missing_robust_public_cv_parity_v1_worker1_robust_as25_aafnet_s2024_f2.log` |
| robustness | AS25_clean aafnet s2024-f3 | cuda:2 | 0 | 114.058 | `outputs/public_followup_probes/logs/remote_missing_robust_public_cv_parity_v1_worker2_robust_as25_aafnet_s2024_f3.log` |
| robustness | AS25_clean aafnet s2024-f4 | cuda:3 | 0 | 114.061 | `outputs/public_followup_probes/logs/remote_missing_robust_public_cv_parity_v1_worker3_robust_as25_aafnet_s2024_f4.log` |
