# module_12_stage4_step23_alert_state_machine

PumpSmart v14.2 — M12 Stage 4 Steps 4.2+4.3 — alert state machine (D2 fix)

- Date: 2026-05-29
- Overall: **FAIL** | BLOCK_M11: True
- score_C: STRONG -> ADVISORY-ONLY

## Transition-matrix gate

| Subgate | Status | Detail |
|---|---|---|
| T1_normal_quiet | PASS | state=NORMAL |
| T2_watch_cusum | PASS | state=WATCH reason=CUSUM S_n 3.000 (gradual-wear path) |
| T3_warn_scoreA | PASS | state=WARN |
| T4_warn_rollmean | FAIL | state=NORMAL rm100=0.0970 drift_ratio=1.000 |
| T5_danger_acute | PASS | state=DANGER |
| T6_danger_mechA | PASS | state=DANGER |
| T7_asym_hysteresis | PASS | held=['DANGER', 'DANGER', 'DANGER', 'DANGER'] after_dwell=WARN |
| T8_no_chatter | PASS | states=['DANGER', 'DANGER', 'DANGER', 'DANGER', 'DANGER', 'DANGER', 'DANGER', 'DANGER', 'DANGER', 'DANGER'] |
| T9_groupC_cap | PASS | state=WARN capped=True |
| T10_groupC_acute_danger | PASS | state=DANGER |
| T11_cavitation_fasttrack | PASS | state=DANGER |
| T12_groupB_phase2 | PASS | state=WARN slope=0.00080 drift_ratio=1.000 |
| T13_scoreC_advisory | PASS | no=NORMAL hi=NORMAL |
| T14_mechC_warn | PASS | state=WARN drift_ratio=1.000 |
| T15_prod_dwell | PASS | clear_dwell=300 |

## Standards basis

- ISA-18.2 / IEC 62682: four states with explicit entry/exit; asymmetric hysteresis (escalate in 1 call, de-escalate after clear_dwell consecutive clear calls); every transition traces to a named trigger (rationalization).
- ISO 13374 / 13379-1: state detection + fault-family logic (cavitation fast-track, Group-C masked cap-to-WARN, Group-B phase-2, label-21 CUSUM path).
- IEC 61511: advisory only — DANGER = immediate manual inspection, never auto-trip.

## Architecture

- New AlertStateMachine reads LOCKED RollingState (theta_t, drift_locked) + CUSUMState (cusum_Sn, cusum_alert) state dicts; owns its own short-horizon score_A buffers (rolling_mean_100/200, slope). ZERO edits to locked classes -> C-25 / Invariant-19 guarantees untouched.
- score_C ADVISORY-ONLY: Step 4.1 found STRONG offline (AUC 0.95) but serve-normal n=0 (no validated live normal baseline), so per ISA-18.2 it cannot drive a high-severity alarm. It annotates reason only.

## Production integration (apply ONLY after reviewing this gate)

- Gate did NOT fully pass; production module NOT emitted. Fix failing subgates first.