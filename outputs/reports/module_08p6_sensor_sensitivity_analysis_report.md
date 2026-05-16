# M8 Patch 6 — Sensor Sensitivity Analysis

**Date:** 2026-05-15  
**Rationale:** Stakeholder review (2026-05) — virtual-sensor sensitivity ceiling check per ISA-37 transducer guidelines.

## What was checked

- Local gain ratio per channel per operating-mode cluster, flagged if p95 > 3.0×.
- Headroom to cluster-conditional winsor ceiling, flagged if < 10%.

## Summary

| Metric | Value |
|---|---|
| rows_processed | 117970 |
| n_gain_records | 32 |
| n_flagged_channel_cluster_pairs | 0 |
| plots_saved | 2 |
| config_saved | True |

## Flagged channel/cluster pairs

None — all channels operating within nominal sensitivity envelope.


## Full gain table

|   cluster_id | cluster_name   | channel      |   gain_median |   gain_p95 |   gain_p99 |   frac_gain_exceeds_3x |   headroom_to_ceiling | headroom_flag   |   ceiling_used |   p99_value |
|-------------:|:---------------|:-------------|--------------:|-----------:|-----------:|-----------------------:|----------------------:|:----------------|---------------:|------------:|
|            0 | cooldown       | X_ACR_Mot.PV |        0      |     0.8852 |     1.3098 |                0       |                0.4576 | False           |         3.0465 |      2.11   |
|            0 | cooldown       | X_ACR_Mot.SV |        0      |     0.0173 |     2.2602 |                0.00276 |                0.1333 | False           |        22.9705 |     20.0426 |
|            0 | cooldown       | X_ACR_Mot.TV |        0      |     0.0226 |     0.3192 |                0       |                0.1468 | False           |         1.0943 |      0.9903 |
|            0 | cooldown       | X_ACR_Pmp.PV |        0      |     0.4853 |     1.0422 |                0.00238 |                0.1219 | False           |         5.108  |      4.6073 |
|            0 | cooldown       | X_ACR_Pmp.SV |        0      |     0.0158 |     0.8549 |                0.00499 |                0.1509 | False           |        31.2149 |     26.6544 |
|            0 | cooldown       | X_ACR_Pmp.TV |        0      |     0.0344 |     0.717  |                0       |                0.1599 | False           |         1.1    |      1      |
|            0 | cooldown       | X_Temp.SV    |        0.0082 |     0.0674 |     0.397  |                0       |                0.1669 | False           |         1.0683 |      0.9487 |
|            0 | cooldown       | X_Pres.SV    |        0.0001 |     0.0469 |     1.1678 |                0       |                0.1157 | False           |         6.0443 |      5.4607 |
|            1 | steady_state   | X_ACR_Mot.PV |        0.2011 |     0.709  |     0.9406 |                0       |                0.3198 | False           |         1.5306 |      1.361  |
|            1 | steady_state   | X_ACR_Mot.SV |        0.2297 |     0.7584 |     1.0682 |                0.00144 |                0.3662 | False           |         1.6585 |      1.4173 |
|            1 | steady_state   | X_ACR_Mot.TV |        0.0054 |     0.0281 |     0.3848 |                0       |                0.2162 | False           |         1.0934 |      0.9876 |
|            1 | steady_state   | X_ACR_Pmp.PV |        0.1578 |     0.4166 |     0.6884 |                0       |                0.4924 | False           |         1.2893 |      1.1468 |
|            1 | steady_state   | X_ACR_Pmp.SV |        0.1073 |     0.3791 |     1.3936 |                0.00704 |                0.2257 | False           |         1.6742 |      1.522  |
|            1 | steady_state   | X_ACR_Pmp.TV |        0.0033 |     0.0269 |     0.4926 |                0       |                0.3652 | False           |         0.965  |      0.8767 |
|            1 | steady_state   | X_Temp.SV    |        0.0084 |     0.0482 |     0.3468 |                0       |                0.2567 | False           |         1.0958 |      0.9955 |
|            1 | steady_state   | X_Pres.SV    |        0.0105 |     0.0669 |     0.6032 |                0       |                0.3663 | False           |         1.3309 |      1.2097 |
|            2 | startup        | X_ACR_Mot.PV |        0      |     0.9792 |     1.278  |                0       |                0.2358 | False           |         2.0824 |      1.8272 |
|            2 | startup        | X_ACR_Mot.SV |        0      |     0.2233 |     0.4199 |                0.00547 |                0.4913 | False           |         1.4426 |      1.2251 |
|            2 | startup        | X_ACR_Mot.TV |        0      |     0.0139 |     0.0506 |                0       |                0.1509 | False           |         1.0947 |      0.9887 |
|            2 | startup        | X_ACR_Pmp.PV |        0      |     0.9388 |     1.322  |                0.00136 |                0.153  | False           |         2.4633 |      2.2394 |
|            2 | startup        | X_ACR_Pmp.SV |        0      |     0.0441 |     0.2241 |                0.00279 |                0.8154 | False           |         1.987  |      1.1822 |
|            2 | startup        | X_ACR_Pmp.TV |        0      |     0.0278 |     0.1874 |                0       |                0.2252 | False           |         1.0779 |      0.971  |
|            2 | startup        | X_Temp.SV    |        0.0053 |     0.0312 |     0.0757 |                0       |                0.1524 | False           |         1.0753 |      0.9604 |
|            2 | startup        | X_Pres.SV    |        0.0012 |     0.0367 |     0.1048 |                0.00311 |                0.2883 | False           |         1.562  |      1.4    |
|            3 | high_load      | X_ACR_Mot.PV |        0.392  |     0.8141 |     1.1488 |                0.00015 |                0.265  | False           |         1.6079 |      1.4468 |
|            3 | high_load      | X_ACR_Mot.SV |        0.1254 |     0.4333 |     0.6323 |                0.00162 |                0.2016 | False           |         2.6548 |      2.3211 |
|            3 | high_load      | X_ACR_Mot.TV |        0.003  |     0.0156 |     0.1989 |                0       |                0.2909 | False           |         0.809  |      0.7347 |
|            3 | high_load      | X_ACR_Pmp.PV |        0.0682 |     0.6129 |     0.8681 |                0       |                0.434  | False           |         1.3633 |      1.2056 |
|            3 | high_load      | X_ACR_Pmp.SV |        0.1097 |     0.6516 |     1.2022 |                0.00293 |                0.2994 | False           |         1.6653 |      1.4661 |
|            3 | high_load      | X_ACR_Pmp.TV |        0.0034 |     0.0201 |     0.0839 |                0       |                0.3508 | False           |         1.096  |      0.9964 |
|            3 | high_load      | X_Temp.SV    |        0.0109 |     0.0311 |     0.0525 |                0.00019 |                0.3336 | False           |         0.7801 |      0.7006 |
|            3 | high_load      | X_Pres.SV    |        0.09   |     0.8831 |     1.3497 |                0.00376 |                0.5847 | False           |         1.2142 |      1.089  |