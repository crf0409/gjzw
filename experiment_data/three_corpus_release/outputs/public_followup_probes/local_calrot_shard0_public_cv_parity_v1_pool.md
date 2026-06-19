# Public Follow-Up Probe Pool

- Label: `local_calrot_shard0_public_cv_parity_v1_pool`
- CV all complete: True
- Devices: `cuda:0, cuda:1, cuda:2, cuda:3`
- Jobs: 26
- Failed jobs: 0

## Setup Steps

| Step | Return code | Seconds |
|---|---:|---:|
| refresh_three_corpus_public_status | 0 | 24.127 |
| public_followup_summary | 0 | 0.179 |
| refresh_three_corpus_public_status | 0 | 19.025 |

## Jobs

| Stage | Cell | Device | Return code | Seconds | Log |
|---|---|---|---:|---:|---|
| calibration_rotation | AS25_clean aafnet s42-f0 | cuda:2 | 0 | 131.562 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker2_calrot_as25_aafnet_s42_f0.log` |
| calibration_rotation | AS25_clean aafnet s42-f2 | cuda:1 | 0 | 131.258 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker1_calrot_as25_aafnet_s42_f2.log` |
| calibration_rotation | AS25_clean aafnet s42-f4 | cuda:3 | 0 | 129.476 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker3_calrot_as25_aafnet_s42_f4.log` |
| calibration_rotation | AS25_clean aafnet s1337-f1 | cuda:0 | 0 | 132.407 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker0_calrot_as25_aafnet_s1337_f1.log` |
| calibration_rotation | AS25_clean aafnet s1337-f3 | cuda:2 | 0 | 131.288 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker2_calrot_as25_aafnet_s1337_f3.log` |
| calibration_rotation | AS25_clean aafnet s2024-f0 | cuda:1 | 0 | 131.228 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker1_calrot_as25_aafnet_s2024_f0.log` |
| calibration_rotation | AS25_clean aafnet s2024-f2 | cuda:3 | 0 | 128.265 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker3_calrot_as25_aafnet_s2024_f2.log` |
| calibration_rotation | AS25_clean aafnet s2024-f4 | cuda:0 | 0 | 127.889 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker0_calrot_as25_aafnet_s2024_f4.log` |
| calibration_rotation | AS25_clean baseline s42-f1 | cuda:3 | 0 | 154.354 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker3_calrot_as25_baseline_s42_f1.log` |
| calibration_rotation | AS25_clean baseline s42-f3 | cuda:3 | 0 | 146.484 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker3_calrot_as25_baseline_s42_f3.log` |
| calibration_rotation | AS25_clean baseline s1337-f0 | cuda:2 | 0 | 145.741 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker2_calrot_as25_baseline_s1337_f0.log` |
| calibration_rotation | AS25_clean baseline s1337-f2 | cuda:0 | 0 | 138.676 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker0_calrot_as25_baseline_s1337_f2.log` |
| calibration_rotation | AS25_clean baseline s1337-f4 | cuda:1 | 0 | 142.659 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker1_calrot_as25_baseline_s1337_f4.log` |
| calibration_rotation | AS25_clean baseline s2024-f1 | cuda:3 | 0 | 130.285 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker3_calrot_as25_baseline_s2024_f1.log` |
| calibration_rotation | AS25_clean baseline s2024-f3 | cuda:0 | 0 | 128.355 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker0_calrot_as25_baseline_s2024_f3.log` |
| calibration_rotation | ASP_clean aafnet s42-f0 | cuda:3 | 0 | 196.827 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker3_calrot_asp_aafnet_s42_f0.log` |
| calibration_rotation | ASP_clean aafnet s42-f2 | cuda:2 | 0 | 194.57 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker2_calrot_asp_aafnet_s42_f2.log` |
| calibration_rotation | ASP_clean aafnet s42-f4 | cuda:1 | 0 | 193.374 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker1_calrot_asp_aafnet_s42_f4.log` |
| calibration_rotation | ASP_clean aafnet s1337-f1 | cuda:0 | 0 | 192.316 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker0_calrot_asp_aafnet_s1337_f1.log` |
| calibration_rotation | ASP_clean aafnet s1337-f3 | cuda:3 | 0 | 194.194 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker3_calrot_asp_aafnet_s1337_f3.log` |
| calibration_rotation | ASP_clean aafnet s2024-f0 | cuda:0 | 0 | 185.215 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker0_calrot_asp_aafnet_s2024_f0.log` |
| calibration_rotation | ASP_clean aafnet s2024-f2 | cuda:1 | 0 | 203.378 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker1_calrot_asp_aafnet_s2024_f2.log` |
| calibration_rotation | ASP_clean aafnet s2024-f4 | cuda:2 | 0 | 179.1 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker2_calrot_asp_aafnet_s2024_f4.log` |
| calibration_rotation | ASP_clean baseline s1337-f4 | cuda:0 | 0 | 196.216 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker0_calrot_asp_baseline_s1337_f4.log` |
| calibration_rotation | ASP_clean baseline s2024-f1 | cuda:1 | 0 | 195.811 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker1_calrot_asp_baseline_s2024_f1.log` |
| calibration_rotation | ASP_clean baseline s2024-f3 | cuda:2 | 0 | 195.759 | `outputs/public_followup_probes/logs/local_calrot_shard0_public_cv_parity_v1_pool_worker2_calrot_asp_baseline_s2024_f3.log` |
