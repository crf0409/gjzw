# Public Follow-Up Probe Pool

- Label: `remote_calrot_shard1_public_cv_parity_v1_pool`
- CV all complete: True
- Devices: `cuda:0, cuda:1, cuda:2, cuda:3`
- Jobs: 25
- Failed jobs: 0

## Setup Steps

| Step | Return code | Seconds |
|---|---:|---:|
| public_robustness_queue | 0 | 0.095 |
| public_calibration_rotation_queue | 0 | 2.176 |
| public_followup_summary | 0 | 0.19 |
| public_robustness_queue | 0 | 0.094 |
| public_calibration_rotation_queue | 0 | 2.134 |

## Jobs

| Stage | Cell | Device | Return code | Seconds | Log |
|---|---|---|---:|---:|---|
| calibration_rotation | AS25_clean aafnet s42-f1 | cuda:0 | 0 | 140.414 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker0_calrot_as25_aafnet_s42_f1.log` |
| calibration_rotation | AS25_clean aafnet s42-f3 | cuda:2 | 0 | 136.854 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker2_calrot_as25_aafnet_s42_f3.log` |
| calibration_rotation | AS25_clean aafnet s1337-f0 | cuda:1 | 0 | 134.712 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker1_calrot_as25_aafnet_s1337_f0.log` |
| calibration_rotation | AS25_clean aafnet s1337-f2 | cuda:3 | 0 | 131.175 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker3_calrot_as25_aafnet_s1337_f2.log` |
| calibration_rotation | AS25_clean aafnet s1337-f4 | cuda:0 | 0 | 137.14 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker0_calrot_as25_aafnet_s1337_f4.log` |
| calibration_rotation | AS25_clean aafnet s2024-f1 | cuda:2 | 0 | 137.019 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker2_calrot_as25_aafnet_s2024_f1.log` |
| calibration_rotation | AS25_clean aafnet s2024-f3 | cuda:1 | 0 | 134.997 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker1_calrot_as25_aafnet_s2024_f3.log` |
| calibration_rotation | AS25_clean baseline s42-f0 | cuda:1 | 0 | 133.056 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker1_calrot_as25_baseline_s42_f0.log` |
| calibration_rotation | AS25_clean baseline s42-f2 | cuda:3 | 0 | 135.842 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker3_calrot_as25_baseline_s42_f2.log` |
| calibration_rotation | AS25_clean baseline s42-f4 | cuda:1 | 0 | 129.973 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker1_calrot_as25_baseline_s42_f4.log` |
| calibration_rotation | AS25_clean baseline s1337-f1 | cuda:3 | 0 | 133.11 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker3_calrot_as25_baseline_s1337_f1.log` |
| calibration_rotation | AS25_clean baseline s1337-f3 | cuda:0 | 0 | 133.739 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker0_calrot_as25_baseline_s1337_f3.log` |
| calibration_rotation | AS25_clean baseline s2024-f0 | cuda:2 | 0 | 130.642 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker2_calrot_as25_baseline_s2024_f0.log` |
| calibration_rotation | AS25_clean baseline s2024-f2 | cuda:1 | 0 | 138.662 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker1_calrot_as25_baseline_s2024_f2.log` |
| calibration_rotation | AS25_clean baseline s2024-f4 | cuda:3 | 0 | 129.809 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker3_calrot_as25_baseline_s2024_f4.log` |
| calibration_rotation | ASP_clean aafnet s42-f1 | cuda:3 | 0 | 164.562 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker3_calrot_asp_aafnet_s42_f1.log` |
| calibration_rotation | ASP_clean aafnet s42-f3 | cuda:2 | 0 | 159.744 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker2_calrot_asp_aafnet_s42_f3.log` |
| calibration_rotation | ASP_clean aafnet s1337-f0 | cuda:0 | 0 | 155.755 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker0_calrot_asp_aafnet_s1337_f0.log` |
| calibration_rotation | ASP_clean aafnet s1337-f2 | cuda:1 | 0 | 156.552 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker1_calrot_asp_aafnet_s1337_f2.log` |
| calibration_rotation | ASP_clean aafnet s1337-f4 | cuda:3 | 0 | 160.686 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker3_calrot_asp_aafnet_s1337_f4.log` |
| calibration_rotation | ASP_clean aafnet s2024-f1 | cuda:0 | 0 | 154.292 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker0_calrot_asp_aafnet_s2024_f1.log` |
| calibration_rotation | ASP_clean aafnet s2024-f3 | cuda:2 | 0 | 165.225 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker2_calrot_asp_aafnet_s2024_f3.log` |
| calibration_rotation | ASP_clean baseline s2024-f0 | cuda:0 | 0 | 158.273 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker0_calrot_asp_baseline_s2024_f0.log` |
| calibration_rotation | ASP_clean baseline s2024-f2 | cuda:1 | 0 | 160.784 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker1_calrot_asp_baseline_s2024_f2.log` |
| calibration_rotation | ASP_clean baseline s2024-f4 | cuda:2 | 0 | 156.294 | `outputs/public_followup_probes/logs/remote_calrot_shard1_public_cv_parity_v1_pool_worker2_calrot_asp_baseline_s2024_f4.log` |
