# -*- coding: utf-8 -*-
"""
STAGE 2 PATCH for app/routers/anomaly.py - APPLY ONLY after green gate matrix.
Carry-forward (Stage 1.5 sec 6): add 22, 23 to GROUP_C_LABELS so the masked-
fault OPERATOR WARNING fires for the dual-channel sensor-failure classes.
Distinct from fault_group_id (22/23 -> group 5).
"""
GROUP_C_LABELS = {13, 14, 15, 16, 17, 22, 23}
