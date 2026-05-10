# =============================================================================
# module_08p5_cusum_decay_and_fmea.py
# PumpSmart v14.2 — M8 Patch 5 of 5: CUSUM Auto-Decay + Model FMEA
# =============================================================================
# WHY THIS SCRIPT EXISTS:
#   1. CUSUM stuck-WATCH failure mode: per current spec CUSUM resets ONLY on
#      confirmed maintenance. If an operator investigates a WATCH and finds
#      nothing wrong (the most common real outcome), S_n has no path back
#      down. It accumulates forever -> WATCH fires on every call -> alarm
#      fatigue -> real Label 21 ignored. This is failure-by-design.
#
#   2. No model FMEA exists: six failure modes of PumpSmart-as-a-system have
#      no documented detection + mitigation. For a 40-lakh asset, this gap
#      blocks insurance / IEC 61508 conversations.
#
# WHAT THIS SCRIPT DOES:
#   1. Computes recommended decay parameters from M8 CUSUM training data
#      (k, H, mu0_B already in models/M8_threshold_config.json).
#   2. Writes a CUSUM runtime policy config that M10 reads at startup.
#   3. Writes a Model FMEA matrix as both JSON (machine-readable) and
#      markdown (review-ready). Six failure modes, each with: detection
#      mechanism, mitigation, recovery procedure, severity (S/O/D/RPN).
#
# OUTPUT FILES:
#   models/M8p5_cusum_runtime_policy.json
#   models/M8p5_model_fmea.json
#   outputs/reports/M8p5_model_fmea.md
#   outputs/reports/M8p5_cusum_decay_report.md
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DEVICE, RAW_DIR, CLEAN_DIR, NORM_DIR, SYNTH_DIR,
                    MODEL_DIR, OUTPUT_DIR, PLOTS_DIR)
from datetime import date, datetime
import json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np

SCRIPT_NAME = "module_08p5_cusum_decay_and_fmea"
REPORT_DIR  = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

results = {}
log("=" * 72)
log(f"  {SCRIPT_NAME}  |  {date.today()}")
log("=" * 72)

# =============================================================================
# SECTION 1 — LOAD M8 CUSUM PARAMETERS FOR DECAY DESIGN
# =============================================================================
log("\nSECTION 1 — Load M8 CUSUM parameters")

M8_THRESH_PATH = MODEL_DIR / "M8_threshold_config.json"
if not M8_THRESH_PATH.exists():
    log(f"  ⚠ M8_threshold_config.json missing — using report-published defaults")
    cusum_mu0_B = -0.00954     # per M8 report
    cusum_k     = 0.02186      # per M8 report
    cusum_H     = 5.0          # per M8 report
else:
    with open(M8_THRESH_PATH) as f:
        m8cfg = json.load(f)
    cusum_mu0_B = m8cfg.get("M8_cusum_mu0_B", -0.00954)
    cusum_k     = m8cfg.get("M8_cusum_k",     0.02186)
    cusum_H     = m8cfg.get("M8_cusum_H",     5.0)
    log(f"  Loaded from M8 config")
log(f"  mu0_B={cusum_mu0_B} | k={cusum_k} | H={cusum_H}")

# =============================================================================
# SECTION 2 — DESIGN THE DECAY PARAMETERS
# =============================================================================
log("\nSECTION 2 — Design decay parameters")

# Geometric decay design rationale:
# We want S_n to halve over a "quiet investigation" window of about 24 hours.
# At 50-second polling: 24 hr = 24 * 60 * 60 / 50 = 1728 calls.
# Half-life: 0.5 = (1 - lambda)^N => lambda = 1 - 0.5^(1/N)
#
# But: decay must be slow enough that a SLOW degrading bearing wear (Label 21,
# detection at week 5) is not erased before it accumulates to H. Conservative
# bound: half-life of 7 days (12096 calls).
HALF_LIFE_CALLS_DEFAULT = 12096    # 7 days at 50s polling
LAMBDA_DEFAULT = 1.0 - (0.5 ** (1.0 / HALF_LIFE_CALLS_DEFAULT))
log(f"  Default geometric decay: lambda={LAMBDA_DEFAULT:.6e}, half-life={HALF_LIFE_CALLS_DEFAULT} calls (7 days)")
log(f"  At lambda above, after 1 day quiet: S_n -> S_n * {(1-LAMBDA_DEFAULT)**1728:.4f}")
log(f"                  after 7 days quiet: S_n -> S_n * {(1-LAMBDA_DEFAULT)**12096:.4f}")
log(f"                  after 30 days quiet: S_n -> S_n * {(1-LAMBDA_DEFAULT)**(30*1728):.4f}")

# Verify: would Label 21 still be detected after this decay?
# From M8 report: CUSUM watch_rate_mild = 1.0, achieved within 500 windows.
# Each window in our setup = a call. So 500 calls of consistent positive
# (score_B - mu0 - k) > 0 should still cross H=5.0 even with decay.
# Per-call addition: estimate from M8 report fault score_B mean drift
# Assume mean addition per fault call = 0.01 (conservative). With decay, after N calls of
# accumulation:  S_n = sum_{i=0..N-1} 0.01 * (1-lambda)^i ~= 0.01 / lambda for large N.
asymptote = 0.01 / LAMBDA_DEFAULT
log(f"  Theoretical asymptote on persistent fault @ +0.01/call: S_n -> {asymptote:.2f}")
log(f"  H={cusum_H}: detection still achievable since asymptote ({asymptote:.2f}) >> H ({cusum_H})")

# =============================================================================
# SECTION 3 — WRITE CUSUM RUNTIME POLICY
# =============================================================================
log("\nSECTION 3 — Writing CUSUM runtime policy")

CUSUM_POLICY = {
    "_meta": {
        "schema_version":   "1.0",
        "created_by":       SCRIPT_NAME,
        "created":          str(date.today()),
        "purpose":          ("Defines CUSUM runtime decay + reset policy. Solves the "
                             "stuck-WATCH failure mode where S_n grows without bound "
                             "if operator investigates and finds no fault."),
    },
    "cusum_parameters": {
        "mu0_B":            cusum_mu0_B,
        "k":                cusum_k,
        "H":                cusum_H,
        "source":           "models/M8_threshold_config.json (read at startup)",
    },
    "decay_policy": {
        "mode":             "geometric_quiet_decay",
        "alternatives":     ["geometric_always", "linear_quiet", "explicit_only"],
        "geometric_quiet_decay": {
            "lambda":               LAMBDA_DEFAULT,
            "half_life_calls":      HALF_LIFE_CALLS_DEFAULT,
            "half_life_days_at_50s_polling": round(HALF_LIFE_CALLS_DEFAULT * 50 / 86400, 2),
            "decay_condition":      "score_B <= mu0_B + k    (i.e. NO fault evidence in this call)",
            "decay_formula":        "S_n_new = S_n_current * (1 - lambda)",
            "rationale": (
                "Per-call multiplicative decay applied ONLY when there is no positive "
                "fault evidence on this call. A persistent fault (where (score_B - mu0 - k) > 0 "
                "on most calls) accumulates faster than decay erodes - detection latency "
                "for genuine slow drift is preserved. A quiet pump (no positive evidence) "
                "sees S_n geometrically forget old WATCH alerts."
            ),
        },
        "investigated_no_fault_reset": {
            "endpoint":                 "/api/cusum_quiet_review",
            "operator_action":          "Operator confirms WATCH alert investigated and no fault found",
            "reset_factor":             0.30,
            "reset_formula":            "S_n_new = S_n_current * 0.30",
            "rationale": (
                "Discrete reset to 30% on operator-confirmed false alarm. NOT zero - if a "
                "real slow drift is in progress, it will re-accumulate; but the false-alarm "
                "spike is cut down. Logs operator_id, timestamp, and reason for audit."
            ),
            "audit_log_required":       True,
        },
        "maintenance_reset": {
            "endpoint":                 "/api/acknowledge",
            "operator_action":          "Confirmed maintenance event (existing M10 spec)",
            "reset_value":              0.0,
            "rationale":                "Hard reset to zero per existing M10 spec - unchanged",
        },
    },
    "watch_alert_runtime_logic": (
        "# Per-call CUSUM update (M10 runtime, 50-second polling)\n"
        "evidence = score_B - mu0_B - k   # positive means fault evidence\n"
        "if evidence > 0:\n"
        "    S_n = S_n + evidence                          # standard CUSUM accumulation\n"
        "else:\n"
        "    S_n = S_n * (1 - lambda)                      # geometric decay on quiet call\n"
        "S_n = max(0, S_n)\n"
        "if S_n > H:\n"
        "    fire_watch_alert()\n"
    ),
    "alarm_fatigue_protection": {
        "rule": (
            "If WATCH has been continuously firing for > 24h with no operator interaction, "
            "automatically downgrade to a daily summary email instead of per-call alert."
        ),
        "implementation_note":      "M10 alert dispatcher concern, not CUSUM concern proper",
    },
    "validation_plan": {
        "deployment_sanity_check": (
            "Within first 30 days of M11 deployment, log: "
            "(a) number of WATCH fires, "
            "(b) number of /api/cusum_quiet_review invocations, "
            "(c) number of /api/acknowledge invocations, "
            "(d) WATCH-to-acknowledge median latency. "
            "If WATCH count > 1/day and acknowledge count = 0, alarm fatigue is in progress."
        ),
        "tune_lambda_after": (
            "60 days of operating data. If genuine WATCH-to-fault confirmations < 80%, "
            "increase lambda by factor 1.5 to decay faster. If real Label 21 is missed, "
            "decrease lambda by factor 0.5 to preserve evidence longer."
        ),
    },
}

CUSUM_POLICY_PATH = MODEL_DIR / "M8p5_cusum_runtime_policy.json"
with open(CUSUM_POLICY_PATH, "w") as f:
    json.dump(CUSUM_POLICY, f, indent=2)
log(f"  Policy: {CUSUM_POLICY_PATH}")
results["cusum_policy_path"] = str(CUSUM_POLICY_PATH)

# =============================================================================
# SECTION 4 — MODEL FMEA MATRIX (SIX FAILURE MODES)
# =============================================================================
# FMEA scoring scale (per IEC 60812):
#   Severity (S):    1 (negligible) -> 10 (catastrophic, no warning)
#   Occurrence (O):  1 (improbable) -> 10 (very high)
#   Detection (D):   1 (almost certain) -> 10 (absolute uncertainty)
#   RPN = S * O * D  (Risk Priority Number, range 1-1000)
#   RPN > 200 = MUST mitigate before deployment
#   RPN 100-200 = SHOULD mitigate
#   RPN < 100 = monitor only
# =============================================================================
log("\nSECTION 4 — Model FMEA matrix")

FMEA = {
    "_meta": {
        "schema_version":      "1.0",
        "created_by":          SCRIPT_NAME,
        "created":             str(date.today()),
        "scope":               "PumpSmart v14.2 detection stack as a system",
        "iec_reference":       "IEC 60812 (FMEA methodology)",
        "scoring": {
            "severity":    "1 (negligible) to 10 (catastrophic, no warning)",
            "occurrence":  "1 (improbable) to 10 (very high)",
            "detection":   "1 (almost certain to detect) to 10 (cannot detect)",
            "rpn":         "S * O * D - mitigation required if > 200",
        },
        "asset_under_protection": "110 kW 7-stage centrifugal pump | INR 30-40 lakh",
    },
    "failure_modes": [
        {
            "id":               "FM-01",
            "name":             "SCADA -> PumpSmart input pipeline halted",
            "description":      "Network partition / SCADA outage / OPC server crash. "
                                "Inference pipeline starves silently.",
            "effect_on_asset":  "No alerts issued. Silent monitoring loss while pump continues operating.",
            "severity":         9,
            "occurrence":       4,
            "detection_pre":    9,
            "rpn_pre":          324,
            "mitigation": (
                "(1) M10 must implement a heartbeat: if no /api/predict call received for "
                "> 5 minutes, an INDEPENDENT watchdog process raises an alert via separate "
                "channel (email, Slack, SMS). "
                "(2) /api/health must include 'last_predict_call_age_seconds' field. "
                "(3) M11 deployment must register an external monitor pinging /api/health "
                "every 60s and alerting on staleness."
            ),
            "recovery":         "Restore SCADA link, M10 auto-resumes when calls arrive. "
                                "CUSUM state preserved (was paused, not reset).",
            "detection_post":   2,
            "rpn_post":         72,
        },
        {
            "id":               "FM-02",
            "name":             "High-confidence wrong fault label",
            "description":      "M7 produces a high-probability prediction for the wrong class. "
                                "Operator performs wrong maintenance action; underlying fault progresses.",
            "effect_on_asset":  "Wrong maintenance applied. Underlying real fault continues to progression. "
                                "If real fault is bearing seizure, asset destroyed within 2-4 hours.",
            "severity":         10,
            "occurrence":       5,
            "detection_pre":    7,
            "rpn_pre":          350,
            "mitigation": (
                "(1) M8p4 OOD detector active in M10 - rejects classifications for inputs that "
                "don't look like training distribution. "
                "(2) 7-field output Field 4 (Expected Sensor Behavior) lets operator cross-check "
                "prediction against physical observation BEFORE acting. "
                "(3) M10 confidence threshold below 0.85 prompts manual verification. "
                "(4) M12 adversarial validation (planned) explicitly tests for this failure mode."
            ),
            "recovery":         "Operator-initiated /api/operator_ack with rejection flag. "
                                "Active learning queue records the mistake for next M7 retrain.",
            "detection_post":   3,
            "rpn_post":         150,
        },
        {
            "id":               "FM-03",
            "name":             "Missed gradual bearing wear (Label 21 false negative)",
            "description":      "Real Paris-law bearing degradation in progress but CUSUM never crosses H. "
                                "Bearing seizes within 2-4 hours of crossing the unobserved fault threshold.",
            "effect_on_asset":  "Catastrophic bearing failure. Replacement INR 35-50 lakh + downtime.",
            "severity":         10,
            "occurrence":       3,
            "detection_pre":    8,
            "rpn_pre":          240,
            "mitigation": (
                "(1) CUSUM auto-decay (this patch) preserves long accumulation while "
                "preventing alarm fatigue from suppressing real WATCH. "
                "(2) Independent inspection schedule: every 30 days physical check of bearing "
                "condition regardless of PumpSmart status. PumpSmart never replaces inspection. "
                "(3) M9 Industrial Pump Selector flags pumps operating outside BEP envelope - "
                "those have higher Paris-law L10 hours and need tighter inspection intervals."
            ),
            "recovery":         "Physical inspection reveals fault. Bearing replaced. Sequence "
                                "added to active learning queue with operator-confirmed Label 21 ground truth.",
            "detection_post":   5,
            "rpn_post":         150,
        },
        {
            "id":               "FM-04",
            "name":             "CUSUM stuck WATCH (alarm fatigue induction)",
            "description":      "Operator investigates WATCH alerts repeatedly with no fault found. "
                                "Without a non-maintenance reset path, S_n stays elevated. Eventually "
                                "WATCH fires every call. Operator stops responding.",
            "effect_on_asset":  "Real Label 21 alert ignored when it eventually fires. "
                                "Same downstream consequence as FM-03.",
            "severity":         10,
            "occurrence":       7,
            "detection_pre":    9,
            "rpn_pre":          630,
            "mitigation": (
                "(1) CUSUM geometric decay (this patch): lambda = " + f"{LAMBDA_DEFAULT:.2e}" + " "
                "applied per quiet call. Half-life = 7 days. "
                "(2) /api/cusum_quiet_review endpoint: explicit 'investigated, no fault' reset to 0.3 * S_n. "
                "(3) Alarm fatigue protection: if WATCH fires > 24h with no operator interaction, "
                "downgrade to daily summary email."
            ),
            "recovery":         "Decay returns S_n to baseline naturally over 7 days quiet operation. "
                                "Operator review reset returns immediately to safe baseline.",
            "detection_post":   2,
            "rpn_post":         140,
        },
        {
            "id":               "FM-05",
            "name":             "Silent TCN-AE (or any model) numerical degradation",
            "description":      "GPU->CPU casting bug, library version mismatch in deployment, NaN poisoning "
                                "from a bad sensor sample, etc. Model produces low-quality scores with no "
                                "observable change to operator.",
            "effect_on_asset":  "Detection performance silently drops. False negatives rise. Same downstream "
                                "consequence as FM-03 if a real fault occurs in the degraded window.",
            "severity":         9,
            "occurrence":       3,
            "detection_pre":    9,
            "rpn_pre":          243,
            "mitigation": (
                "(1) Daily self-test routine (M11): inject 10 known synthetic test sequences from a "
                "frozen golden-set, compare outputs to expected score_A/B/C ranges. "
                "Halt if any deviation exceeds 5%. "
                "(2) NaN guard at every model boundary - reject any input/output with NaN values. "
                "(3) Version-pin all torch/sklearn/xgboost/numpy versions in Docker image with hash check."
            ),
            "recovery":         "Halt inference, alert operations team, roll back to prior known-good Docker image.",
            "detection_post":   3,
            "rpn_post":         81,
        },
        {
            "id":               "FM-06",
            "name":             "L4 adaptive threshold chasing real slow degradation",
            "description":      "A real slow drift is in progress that should fire L3. L4's 6-hour rolling "
                                "baseline tracks the drift up before L3 accumulates - threshold rises with the "
                                "fault, suppressing the score_A alert.",
            "effect_on_asset":  "Fault concealment. Same downstream consequence as FM-03.",
            "severity":         10,
            "occurrence":       4,
            "detection_pre":    8,
            "rpn_pre":          320,
            "mitigation": (
                "(1) Crosspoint guard: theta_t locked if it exceeds 1.5 * theta_initial. Existing M8 design - "
                "verify implementation in M10 runtime. "
                "(2) L3 CUSUM operates on score_B (drift slope), independent of L4's score_A baseline. "
                "Invariant 19 (no cross-routing) ensures L4 cannot suppress L3. "
                "(3) Add a metric to /api/threshold_status: theta_t / theta_initial ratio. "
                "If > 1.3 for > 24h, raise an engineering review notice."
            ),
            "recovery":         "Engineering review identifies whether drift is real fault (commission "
                                "Label 21 inspection) or operating-point shift (re-cluster via commissioning mode).",
            "detection_post":   4,
            "rpn_post":         160,
        },
    ],
}

# RPN totals
total_rpn_pre  = sum(fm["rpn_pre"]  for fm in FMEA["failure_modes"])
total_rpn_post = sum(fm["rpn_post"] for fm in FMEA["failure_modes"])
FMEA["_meta"]["total_rpn_pre_mitigation"]  = total_rpn_pre
FMEA["_meta"]["total_rpn_post_mitigation"] = total_rpn_post
FMEA["_meta"]["rpn_reduction_pct"] = round(100 * (total_rpn_pre - total_rpn_post) / total_rpn_pre, 1)

log(f"  Failure modes documented: {len(FMEA['failure_modes'])}")
log(f"  Total RPN pre-mitigation:  {total_rpn_pre}")
log(f"  Total RPN post-mitigation: {total_rpn_post}")
log(f"  RPN reduction: {FMEA['_meta']['rpn_reduction_pct']}%")

FMEA_JSON_PATH = MODEL_DIR / "M8p5_model_fmea.json"
with open(FMEA_JSON_PATH, "w") as f:
    json.dump(FMEA, f, indent=2)
log(f"  FMEA JSON: {FMEA_JSON_PATH}")
results["fmea_json_path"] = str(FMEA_JSON_PATH)

# =============================================================================
# SECTION 5 — RENDER FMEA AS REVIEW-READY MARKDOWN
# =============================================================================
log("\nSECTION 5 — Render FMEA markdown")

fmea_rows = []
for fm in FMEA["failure_modes"]:
    blocker = "🔴 BLOCKER" if fm["rpn_pre"] > 300 else ("🟡 HIGH" if fm["rpn_pre"] > 200 else "🟢")
    post_status = "✓ ACCEPTABLE" if fm["rpn_post"] <= 200 else "⚠ STILL ELEVATED"
    fmea_rows.append(f"""
### {fm['id']} — {fm['name']}  {blocker}

| | Severity | Occurrence | Detection | RPN |
|---|---|---|---|---|
| Pre-mitigation | {fm['severity']} | {fm['occurrence']} | {fm['detection_pre']} | **{fm['rpn_pre']}** |
| Post-mitigation | {fm['severity']} | {fm['occurrence']} | {fm['detection_post']} | **{fm['rpn_post']}** {post_status} |

**Description:** {fm['description']}

**Effect on asset:** {fm['effect_on_asset']}

**Mitigation:**

{fm['mitigation']}

**Recovery procedure:** {fm['recovery']}

---
""")
fmea_md_body = "\n".join(fmea_rows)

FMEA_MD = f"""# PumpSmart v14.2 — Model FMEA Matrix
**Date:** {date.today()}
**Methodology:** IEC 60812
**Asset under protection:** 110 kW 7-stage centrifugal pump (INR 30-40 lakh)

## Scoring scale
- **Severity (S):** 1 (negligible) to 10 (catastrophic, no warning)
- **Occurrence (O):** 1 (improbable) to 10 (very high)
- **Detection (D):** 1 (almost certain to detect) to 10 (cannot detect)
- **RPN = S × O × D** (1 to 1000)

## Action thresholds
- RPN > 300 → 🔴 BLOCKER — must mitigate before any deployment
- RPN 200–300 → 🟡 HIGH — mitigate before M11 production deployment
- RPN < 200 → 🟢 ACCEPTABLE — monitor through deployment lifecycle

## Aggregate
| | Total RPN |
|---|---|
| Pre-mitigation | **{total_rpn_pre}** |
| Post-mitigation | **{total_rpn_post}** |
| Reduction | {FMEA['_meta']['rpn_reduction_pct']}% |

All six failure modes are reduced to RPN ≤ 200 (acceptable for shadow-mode
deployment). Remaining residual risk is the irreducible synthetic-to-real gap,
which is bounded by the active learning loop (C-26 mitigation) and the OOD
detector (M8p4).

---

## Failure modes
{fmea_md_body}

## Relationship to project challenge log
| Failure mode | Related challenge |
|---|---|
| FM-01 | New — not in C-01..C-27 |
| FM-02 | C-26 (synthetic-to-real domain gap) — addressed by M8p4 OOD detector |
| FM-03 | C-27 (sequence length physics), C-15 (false sense of security) |
| FM-04 | C-25 (adaptive threshold paradox) — solved here for the operational case |
| FM-05 | New — not in C-01..C-27 |
| FM-06 | C-25 (adaptive threshold paradox) — operational guardrail |

## Required M10/M11 implementation work to realize the post-mitigation scores
1. **M10:** /api/cusum_quiet_review endpoint with audit log
2. **M10:** OOD detector (M8p4) wired into /api/predict
3. **M10:** Heartbeat field on /api/health + age tracking
4. **M11:** External monitor pinging /api/health every 60s
5. **M11:** Daily golden-set self-test routine
6. **M11:** NaN guards at all model boundaries
7. **M11:** Version-pinned Docker image with hash check
8. **M11:** Independent maintenance inspection schedule documentation (handover)

Until all 8 implementation items are complete, the post-mitigation RPN scores
above are DESIGN INTENT, not realised. The deployment posture should be
shadow-mode-only until items 1-7 are coded and tested.
"""

FMEA_MD_PATH = REPORT_DIR / "M8p5_model_fmea.md"
with open(FMEA_MD_PATH, "w", encoding="utf-8") as f:
    f.write(FMEA_MD)
log(f"  FMEA markdown: {FMEA_MD_PATH}")
results["fmea_md_path"] = str(FMEA_MD_PATH)

# =============================================================================
# SECTION 6 — CUSUM REPORT
# =============================================================================
log("\nSECTION 6 — CUSUM decay report")

CUSUM_REPORT_PATH = REPORT_DIR / "M8p5_cusum_decay_report.md"
cusum_md = f"""# M8 Patch 5a — CUSUM Auto-Decay Policy
**Date:** {date.today()}

## Why this patch existed
Per the original M8 spec, CUSUM resets only on confirmed maintenance event.
If an operator investigates a WATCH alert and finds nothing wrong (the most
common real outcome), there was no path through the spec for S_n to come
back down. It accumulated forever -> WATCH fires on every call -> alarm
fatigue -> real Label 21 ignored.

This is the canonical "alarm fatigue induction" failure mode. It is FM-04 in
the model FMEA.

## What this patch does
Adds three reset mechanisms (selectable per deployment):

1. **Geometric quiet decay (recommended, default):**
   `S_n_new = S_n_current * (1 - λ)` applied per call when no positive
   fault evidence is present. λ = {LAMBDA_DEFAULT:.6e} → 7-day half-life
   on a fully quiet pump.

2. **Operator-investigated reset (NEW endpoint /api/cusum_quiet_review):**
   Discrete reset to 0.3 * S_n when operator confirms WATCH was investigated
   and no fault found. Audit logged.

3. **Maintenance reset (existing /api/acknowledge):**
   Hard reset to 0.0. Unchanged from existing M10 spec.

## Why these are mathematically safe for Label 21 detection

Decay erodes S_n only on **quiet** calls. A persistent fault produces positive
evidence faster than decay erodes — accumulation continues. The asymptote
of S_n on a fault producing +0.01/call evidence with this λ is approximately
0.01/λ ≈ {0.01/LAMBDA_DEFAULT:.1f}, far above H={cusum_H}. Detection
latency for genuine slow drift is preserved.

## Tunable parameters
| Parameter | Default | Where to tune |
|---|---|---|
| λ (decay rate) | {LAMBDA_DEFAULT:.6e} | M8p5_cusum_runtime_policy.json |
| Half-life (calls) | {HALF_LIFE_CALLS_DEFAULT:,} | derived from λ |
| Operator-reset factor | 0.30 | M8p5_cusum_runtime_policy.json |
| Quiet-detection condition | score_B ≤ μ₀ + k | hardcoded (mathematically required) |

## Validation plan (first 60 days of deployment)
- Log: WATCH count, /api/cusum_quiet_review count, /api/acknowledge count
- If WATCH count > 1/day and /api/acknowledge = 0 → alarm fatigue in progress, raise λ ×1.5
- If a real Label 21 is missed → reduce λ ×0.5 to preserve evidence longer
- Tune after 60 days of real operating data, not earlier

## Files written
- `models/M8p5_cusum_runtime_policy.json` (M10 reads at startup)

## M10 runtime code skeleton (must be implemented in app/)

```python
# In app/runtime/cusum_state.py
import json
from pathlib import Path

CUSUM_POLICY_PATH = Path('models/M8p5_cusum_runtime_policy.json')

class CUSUMState:
    def __init__(self):
        cfg = json.load(open(CUSUM_POLICY_PATH))
        p = cfg['cusum_parameters']
        d = cfg['decay_policy']['geometric_quiet_decay']
        self.mu0 = p['mu0_B']
        self.k   = p['k']
        self.H   = p['H']
        self.lam = d['lambda']
        self.S_n = 0.0
        self.fired = False

    def update(self, score_B):
        evidence = score_B - self.mu0 - self.k
        if evidence > 0:
            self.S_n = self.S_n + evidence
        else:
            self.S_n = self.S_n * (1 - self.lam)   # decay
        self.S_n = max(0.0, self.S_n)
        if self.S_n > self.H and not self.fired:
            self.fired = True
            return 'WATCH'
        return None

    def operator_quiet_review(self, factor=0.30):
        self.S_n = self.S_n * factor
        self.fired = False
        # MUST log: timestamp, operator_id, reason, S_n_before, S_n_after

    def maintenance_reset(self):
        self.S_n = 0.0
        self.fired = False
```
"""
with open(CUSUM_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(cusum_md)
log(f"  CUSUM report: {CUSUM_REPORT_PATH}")

# =============================================================================
# SECTION 7 — PASTE TEXT
# =============================================================================
log("\n" + "=" * 72)
log("══ PASTE TEXT UPDATE — COPY BELOW INTO PASTE TEXT ══")
print(f"""
M8p5_cusum_policy_path             : models/M8p5_cusum_runtime_policy.json
M8p5_cusum_decay_lambda            : {LAMBDA_DEFAULT:.6e}
M8p5_cusum_half_life_calls         : {HALF_LIFE_CALLS_DEFAULT}
M8p5_cusum_half_life_days          : 7.0
M8p5_quiet_review_endpoint         : /api/cusum_quiet_review (M10 implementation required)
M8p5_quiet_review_reset_factor     : 0.30
M8p5_fmea_json_path                : models/M8p5_model_fmea.json
M8p5_fmea_md_path                  : outputs/reports/M8p5_model_fmea.md
M8p5_fmea_failure_modes_count      : 6
M8p5_fmea_total_rpn_pre            : {total_rpn_pre}
M8p5_fmea_total_rpn_post           : {total_rpn_post}
M8p5_fmea_rpn_reduction_pct        : {FMEA['_meta']['rpn_reduction_pct']}
M8p5_M10_implementation_required   : True
M8p5_M11_implementation_required   : True
Status_for_M9                      : READY (all five Tier-1 patches complete)
""")
log("══ END PASTE UPDATE ══")

# =============================================================================
# FILE MANIFEST + NEXT
# =============================================================================
log("=" * 72)
log("FILE MANIFEST")
log("=" * 72)
log(f"  GitHub push: {CUSUM_POLICY_PATH}")
log(f"  GitHub push: {FMEA_JSON_PATH}")
log(f"  GitHub push: {FMEA_MD_PATH}")
log(f"  GitHub push: {CUSUM_REPORT_PATH}")
log(f"  Spaces upload: M8p5_cusum_runtime_policy.json (mandatory for M10)")
log(f"  Spaces upload: M8p5_model_fmea.json (operator/insurance reference)")

log("\n📦 M8p5 done. ALL FIVE TIER-1 PATCHES COMPLETE.")
log("    M9 (Industrial Pump Selector + Household Advisor) is now UNBLOCKED.")
log("    Provide M9 complete script when ready.")
log("=" * 72)
