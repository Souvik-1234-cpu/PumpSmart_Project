# PumpSmart v14.2 — M9 / M10 / M11 Context Document
**Architecture: v14.2 + M8p6 Sensor Sensitivity Guardrail**
**Document status: CURRENT — Supersedes `Sequential_Industry_grade_verification_pathway.md`**
**Last updated: May 2026 | Audit basis: Industrial Audit v5.0**

---

## STATUS SUMMARY — ALL MODULES

| Module | Name | Status | Notes |
|---|---|---|---|
| M1 | Cleaning & Segmentation | ✅ LOCKED | |
| M2 | EDA & K-Means Clustering | ✅ LOCKED | 4 clusters |
| M3 | Physics-Dimensionless Normalisation | ✅ LOCKED | M3_normalization_config.json |
| M4 | LSTM-AE Baseline (L1) | ✅ LOCKED | q = 0.110058 FIXED |
| M5 | Physics Engine & Validation | ✅ LOCKED | 20/20 nameplate pass |
| M6A | Synthetic Generator Group A | ✅ LOCKED | |
| M6B | Synthetic Generator Groups B–E | ✅ LOCKED | ~31,800 sequences total |
| M6.5r | Feature Retrain (z_t feats) | ✅ LOCKED | 35-feature schema |
| M7 | XGBoost Classifier + SHAP | ✅ LOCKED | Seq-level F1 = 0.9965 ± 0.0005 |
| M8 | TCN-AE + CUSUM + Adaptive Threshold | ✅ LOCKED | All Tier-1.5 items closed |
| **M9** | **Industrial Pump Selector** | **✅ LOCKED** | **24/24 gates passing** |
| **M10** | **FastAPI Application Layer** | **🔵 ACTIVE** | **Phase 1 next** |
| M11 | Docker + Hugging Face Deploy | ⏳ BLOCKED | Requires M12 first (T2-1) |
| M12 | Testing + Active Learning Loop | ⏳ PENDING | Must precede M11 |

---

## MODULE 9 — COMPLETE AND LOCKED

**M9: Industrial Pump Selector**
- Status: 24/24 gates passing (three patch iterations)
- Output: Physics-only industrial pump sizing engine
- Key routing rule (T2-3, locked):

```python
def route_pump(power_kW, head_m, stages, pressure_bar):
    is_industrial = (power_kW >= 30.0 and head_m >= 80.0 and
                     stages >= 3 and pressure_bar >= 8.0)
    if is_industrial:
        return 'industrial_ml_pipeline'
    elif power_kW <= 5.0 and stages == 1 and pressure_bar <= 5.0:
        return 'household_physics_advisory'
    else:
        return 'OUT_OF_SCOPE'  # 5–30 kW gap — neither path safe
```

- Mandatory guard in all inference code:
```python
if pump_type == 'household':
    return physics_advisory_only()
else:
    return ml_prediction()
```

---

## MODULE 10 — ACTIVE

### Technology Stack (LOCKED)

| Layer | Technology |
|---|---|
| Backend framework | FastAPI + uvicorn (ASGI) |
| ML inference | PyTorch 2.6, map_location='cpu' for deploy |
| Fault classifier | XGBoost 22-class, CPU deploy |
| Frontend | React JSX via Jinja2 templates |
| Charts | Chart.js v4.4 CDN |
| Deployment target | Hugging Face Spaces (Docker container) |

### UI Structure (LOCKED — Mockup confirmed May 2026)

**Landing page:** Two entry cards — Household Pump Advisor (navy/sky blue) + Industrial Sensor Monitor (cyan/lighter blue). Dark navy background, radial blue ambient glow.

**Acknowledgment gate:** Hard gate before dashboard. 6 acknowledgment points. Padlock + 40% dim on all non-guide tabs before acknowledgment. Acknowledgment also available inline at bottom of every Guide section. Re-acknowledge button in topbar for new operators.

**Industrial Dashboard — Left sidebar navigation, 7 tabs:**
1. Dashboard
2. Sensor Plugin
3. Analytics
4. Predictions
5. History
6. Settings
7. Guide & Disclaimer

### 8 API Routes (LOCKED)

| Route | Method | Purpose |
|---|---|---|
| GET /health | GET | CUSUM S_n, theta_t, z_t buffer, commissioning mode — polled every 30s |
| POST /api/anomaly_detect | POST | Main inference: 4 layers + 7-field output + M8p6 addendum if triggered |
| POST /api/classify_fault | POST | 22-class XGBoost + causal chain for Group B |
| POST /api/select_pump | POST | M9 physics-only industrial selector |
| GET /api/household | GET | Physics-only, zero ML, advisory_disclaimer always present |
| POST /api/acknowledge | POST | **Operational reset ONLY** — resets CUSUM S_n, z_t buffer, rolling baseline + logs timestamp. Does NOT write to active-learning data. |
| GET /api/validate_model | GET | SHA-256 hash verification all model files |
| GET /api/physics_context | GET | Static 22-label plain-language lookup |

### FastAPI Lifespan Startup Artifacts (ALL required — hard fail if any missing)

| Artifact | Purpose |
|---|---|
| models/M4_lstm_ae_state_dict.pt | L1 LSTM-AE, map_location='cpu' |
| models/M7_xgboost.json | 22-class classifier, CPU deploy |
| models/M4_threshold_config.json | q = 0.110058, LOCKED |
| models/M8_threshold_config.json | Cluster-conditional + score_C thresholds |
| models/M8p6_sensor_sensitivity_config.json | ISA-37 gain limits + headroom flags (C-28) |
| models/fault_rules_v3.json | 22-class label map for 7-field output |
| models/M3_normalization_config.json | Cluster baselines for inference normalisation |
| outputs/M2_cluster_bounds.csv | Cluster ceiling reference |

### 7-Field Mandatory Output (NEVER REDUCE — Invariant)

| Field | Name | Content |
|---|---|---|
| 01 | Primary Classification | Specific fault class name from 22-class M7 set |
| 02 | Probability Matrix | XGBoost confidence %. Flags UNKNOWN FAULT if <70% |
| 03 | Physical Interpretation | Plain-language: what is physically happening inside the pump |
| 04 | Expected Signature | What sensors should do if prediction is correct |
| 05 | Consequence Horizon | Timeline if fault ignored — hours/days/weeks |
| 06 | Action Protocol | Specific sequenced maintenance steps. M8p6 addendum sub-line appended in amber at 0.85em if sensor ceiling-approach triggers. Override_existing_prediction: false (locked). |
| 07 | Model Disclaimer | "Trained on CIRA-anchored physics-synthetic data for 110 kW 7-stage pump at 2980 RPM, 40 bar. Advisory only. Verify physically. Confidence 0.65–0.85 expected on real faults (C-26)." |

### Alert State Machine (LOCKED)

| State | Trigger | UI Action |
|---|---|---|
| NORMAL | score_A < θ_t AND CUSUM S_n < 2.0 | Routine monitoring |
| WATCH | CUSUM S_n ≥ 2.0, score_A still below θ_t | Event log only — NO popup |
| WARN | score_A ≥ θ_t | Full overlay popup |
| DANGER | score_A ≥ 1.5 × θ_t | Full overlay popup + pulsing red |

**Popup behaviour (WARN / DANGER):**
- Acknowledge button: **operational reset only** — resets CUSUM S_n, z_t buffer, rolling baseline. Does NOT write active-learning row.
- Dismiss button: closes popup only. Alert remains active in log and dashboard.
- Popup text: *"Acknowledge this alert to reset the alarm state. To rate the model's accuracy, please use the Predictions tab once you have physically verified the fault."*

**Predictions tab verification buttons:** Correct / Incorrect / Unsure — these write ONE row to the active-learning data store (HF Datasets API). This is the ONLY write point.

### Sensor Interruption States (3 Cases)

1. Some sensors disconnected, pump running → warning banner, model partially paused on affected channels, sidebar count turns amber
2. All sensors disconnected → modal: "Has the pump been shut down?" Yes = model sleeps, auto-resumes on return. No = critical alert.
3. Individual sensor gradual failure → red dot on card, monitoring on remaining channels. 2+ simultaneous failures = model reliability significantly reduced.

### Sensor Plugin — Per-Cluster Ranges (v5.0-A Concern 3)

Sensor Plugin tab displays **per-cluster ranges**, not steady-state-only ranges. The active cluster is highlighted. This prevents operators misreading startup/cooldown values as out-of-spec.

| Plain English Name | Ch ID | Unit | Startup (C0) | Steady-state (C1) | High-load (C2) | Cooldown (C3) |
|---|---|---|---|---|---|---|
| Motor Vibration (RMS) | Mot.SV | mm/s | 7–14 | 3.5–5.5 | 4.5–6.5 | 2.5–4.5 |
| Pump Vibration (RMS) | Pmp.SV | mm/s | 6–12 | 3.0–5.0 | 4.0–6.0 | 2.0–4.0 |
| Motor Winding Temperature | Mot.TV | °C | 65–80 | 60–75 | 70–82 | 58–72 |
| Pump Discharge Pressure | Pmp.PV | bar | 0–41 | 38–41 | 39–42 | 10–38 |
| Bearing Housing Temperature | Temp.SV | °C | 60–75 | 55–70 | 65–78 | 50–65 |
| Suction Side Pressure | Pres.SV | bar | 0.5–3.5 | 1.5–3.5 | 1.0–3.0 | 0.3–2.5 |
| Pump Casing Temperature | Pmp.TV | °C | 55–70 | 50–65 | 60–72 | 35–55* |
| Motor Power Draw | Mot.PV | kW | 60–115 | 100–115 | 108–120 | 30–100 |

*Cooldown Pmp.TV may drop below ambient by up to 1.4°C due to flash evaporation at 40 bar shutdown — correct physics (C-09/C-10), not a sensor fault.

*Note: Startup cluster vibration 2–4× higher than steady-state = correct physics (shaft resonance during acceleration, not a fault).*

### M8p6 Sensor Sensitivity Guardrail (C-28, Principle 14 — LOCKED)

- Config: `models/M8p6_sensor_sensitivity_config.json` loaded at lifespan startup
- At every inference call: compute live gain_p95 and headroom for active cluster
- If either crosses flag threshold → Field 6 receives amber addendum sub-line (0.85em, #e67e22)
- **Critical: addendum NEVER overrides fault label or confidence. override_existing_prediction: false (locked).**
- Two CIRA borderline channels (expected to trigger on non-CIRA pumps by design):
  - Pres.SV: headroom 11.6%, borderline (2.0× high-load ceiling tightest in system per C-18)
  - Pmp.PV: headroom 12.2%, borderline (startup ceiling 3.2× per C-17 ISO 13373-3)

### Score Routing — Invariant 19 (LOCKED, CRITICAL)

```
score_A → L4 Adaptive Threshold rolling baseline ONLY
score_B → L3 CUSUM ONLY
score_C → M7 XGBoost ONLY
```
Never cross-route these. Prevents score_A operating-point shifts from generating false Label 21 CUSUM alarms.

### Risk Gauge Needle Behaviour (v5.0 Concern 4 — IMPLEMENTED)

- **Within-state movement:** smooth 0.08 interpolation factor per frame (requestAnimationFrame)
- **State-change transitions (NORMAL↔WATCH↔WARN↔DANGER):** needle SNAPS immediately to new zone — no interpolation lag. Consistent with NUREG-0700 safety-critical display guidance.

### Sidebar Pump Panel Badge (v5.0-D — IMPLEMENTED)

`PUMP-0032 (single-pump v14.2) | 110 kW · 7-stage · 40 bar`

### Active Learning — Data Architecture (v5.0-B — LOCKED SCHEMA)

**Persistence layer:** HF Datasets API — NOT local disk.
- Reason: HF free tier has zero persistent storage. Any CSV written to /tmp or /data is wiped on every restart.
- Pattern: `push_learning_row(row_dict)` helper in backend, called from operator verdict endpoint only.
- Each row pushed as JSON file to companion `pumpsmart-active-learning` dataset repo.
- Free tier supports ~5–10 million events.

**27-column schema v1.0 (LOCKED):**

| Column | Type | Source |
|---|---|---|
| timestamp_utc | ISO 8601 | server clock |
| pump_id | string | sidebar |
| prediction_id | UUID | server |
| cluster_id | int 0–3 | M2 K-Means |
| predicted_label_int | int 0–21 | M7 |
| predicted_label_name | string | fault_rules_v3.json |
| confidence_pct | float | M7 softmax |
| score_A | float | L1 |
| score_B | float | L2 |
| score_C | float | L2 |
| cusum_s_n | float | L3 |
| theta_t | float | L4 |
| alert_state | enum | UI — NORMAL/WATCH/WARN/DANGER |
| m8p6_sensor_flag | bool | M8p6 |
| m8p6_flagged_channels | csv-string | M8p6 |
| mahal_dist | float | T1.4 OOD |
| ood_flag | bool | T1.4 |
| raw_sensor_window | json | 50×8 sensor array at fault time |
| top_3_shap_features | json | Analytics tab SHAP values |
| operator_verdict | enum | Predictions tab — CORRECT/INCORRECT/UNSURE/PENDING |
| operator_correct_label | int 0–21 | Optional — if INCORRECT |
| verdict_timestamp_utc | ISO 8601 | server |
| time_to_verdict_seconds | int | Calculated: verdict_ts - prediction_ts |
| physical_inspection_done | bool | Optional operator field |
| inspection_notes | string | Optional free-text |
| data_source | enum | M12_SYNTHETIC / SHADOW_REAL / PRODUCTION_REAL |
| consent_granted_by | string | GDPR — operator ID, for future cross-plant aggregation |

**Write rules:**
- `/api/acknowledge` (popup button) → does NOT write schema row (operational reset only)
- Predictions tab Correct/Incorrect/Unsure → writes ONE row per response
- `operator_verdict` defaults to PENDING until operator responds

### Design Language

**Industrial monitor palette:**
- Background: #04101e (deep navy) + radial blue ambient glow
- Card surfaces: rgba(6,18,36,0.88) + backdrop blur (glassmorphism)
- Primary accent: #00d4ff (cyan)
- NORMAL: #00e676 | WATCH: #ffcc00 | WARN: #ff8800 | DANGER: #ff2244 (pulsing)
- M8p6 addendum: #e67e22 (amber, distinct from WARN orange)

**Animations:**
- Card entry: fadeSlide (opacity 0→1, translateY 12px→0, staggered delays)
- Risk gauge needle: 0.08 factor per frame within state; immediate snap on state change
- Arc gauge fill: 0.12 factor per frame
- Alert state: CSS transition 0.4s ease
- Fault popup: scaleIn 0.3s cubic-bezier

### M10 Implementation Phases

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | main.py — FastAPI entry point, lifespan model loading, router registration, UI template routes with placeholder responses | 🔵 NEXT |
| Phase 2 | Inference pipeline — anomaly_detect route with full 4-layer stack | ⏳ |
| Phase 3 | Frontend React JSX integration + Chart.js wiring | ⏳ |
| Phase 4 | HF Datasets API persistence + active learning endpoint | ⏳ |
| Phase 5 | End-to-end integration test (7-field output, alert state machine, M8p6 addendum) | ⏳ |

---

## MODULE 11 — BLOCKED

**M11: Docker + Hugging Face Deploy**

**Status: BLOCKED** — must not begin until M12 adversarial validation passes.

**Reason (T2-1):** Deploying before M12 means publishing a model whose generalisation on novel-distribution sequences is unmeasured. If M12 reveals failure, a retraction or patch on a live deployed system is required. M12 must come first.

**Gate to pass before M11:** `T2-1_M12_before_M11_PASS`
- Block M11 if M12 macro detection rate < 80% on adversarial set

**Deployment platform analysis (v5.0, Section 13):**

| Resource | HF Free Tier | PumpSmart Fit |
|---|---|---|
| RAM | 16 GB | ✅ M4+M7+M8 ≈ 200–500 MB total |
| CPU | 2 vCPU | ✅ Requires T2-2 benchmark to confirm <5s per inference |
| Ephemeral disk | 50 GB | ✅ Full repo well under |
| Git repo limit | 1 GB | ⚠️ Tight — model PKLs tracked via Git LFS |
| Persistent storage | NONE | 🔴 BLOCKER for CSV — solved by HF Datasets API (v5.0-C) |

**CPU inference benchmark (T2-2) — must confirm before M11:**
- M4 LSTM-AE forward pass (one window)
- M8 TCN-AE forward pass (one z_t buffer)
- M8p4 OOD detector (one Mahalanobis distance)
- M7 XGBoost predict_proba (one feature row)
- M8p5 CUSUM update (one score_B value)
- 7-field output rendering
- **Gate: total < 5 seconds per full inference cycle on 2-vCPU constrained environment**
- Fallback if fails: ONNX export of LSTM-AE and TCN-AE (typically 3–5× faster than PyTorch eager on CPU)

---

## MODULE 12 — PENDING (runs BEFORE M11)

**M12: Adversarial Testing + Active Learning Loop**

- Generates fresh physics-synthetic sequences via M5 engine with parameters NOT used in M6 training (different severity values, cluster contexts, lag combinations within physics-allowed envelope)
- True held-out evaluation — model has never seen these
- Required detection chain per scenario: score_A rises → score_B slope positive → CUSUM accumulates → correct alert state → 7-field output complete
- Pass gate: detection latency on held-out ≤ 1.5× training-set latency
- **Block M11 if M12 macro detection rate < 80%**

**Active learning loop (T3-2):**
1. Operator verdict → HF Datasets API row
2. After 50 real confirmed faults (target: ~60 days shadow ops): trigger M7 retrain
3. Retrain uses 80% synthetic + 20% real — do NOT discard synthetic (covers rare fault classes)
4. New model deployed only after M12 adversarial re-pass on combined dataset

---

## DEFERRED ITEMS (Tier 2 — Before M11)

| Item | Description | Blocking? |
|---|---|---|
| T2-1 | M12 before M11 | Yes — M11 blocked until M12 passes |
| T2-2 | CPU inference benchmark (<5s gate) | Yes — before HF deploy |
| T2-3 | Physical-parameter routing (locked in M9) | ✅ Done |
| T2-4 | Baseline LSTM-AE comparison (publishable benchmark) | No — publication enhancement |
| T2-5 | Threshold sensitivity audit (FPR-vs-threshold sweep) | No — deferred |
| T2-6 | Config drift hash registry | Before M11 |
| T2-7 | Cluster assignment hysteresis | Before M11 |
| T2-9 | Group B v1↔v2 cross-evaluation | Blocked by 32-vs-33 feature column mismatch |

## DEFERRED ITEMS (Tier 3 — First 60 Days Shadow)

| Item | Description |
|---|---|
| T3-1 | Real-world FPR documentation (30-day shadow protocol) |
| T3-2 | Active learning queue + first retrain (50+ confirmed real faults) |
| T3-3 | Input distribution drift monitoring (PSI per channel, monthly) |
| T3-4 | IEC 61508 SIL documentation (only if pursuing certification) |
| T3-5 | Cross-pump generalisation study (fleet extension) |

---

## AUDIT STATUS — v5.0 (May 2026)

**Industrial Audit v5.0 — all prior items closed:**
- Tier-1 (8 items): CLOSED
- Tier-1.5 (3 items): CLOSED
- M9 + M10 UI mockup audit: COMPLETE (5 concerns reviewed, all resolved)
- HF deployment capacity: ANALYSED (inference yes, persistence solution = HF Datasets API)

**v5.0 Action Items — implementation status:**
| Item | Description | Status |
|---|---|---|
| v5.0-A | Split acknowledgment: popup = operational reset, Predictions tab = active learning write | ✅ Implemented in JSX |
| v5.0-B | Lock 27-column active-learning schema v1.0 | ✅ Locked above |
| v5.0-C | HF Datasets API persistence (not local disk) | ✅ Architecture locked |
| v5.0-D | "(single-pump v14.2)" badge in sidebar pump panel | ✅ Implemented in JSX |

**UI concerns resolved in mockup:**
| Concern | Resolution |
|---|---|
| Concern 3 — Sensor nominal ranges | Per-cluster ranges with active cluster highlighted (Option A) |
| Concern 4 — Gauge needle lag on state change | Immediate snap on state transition; smooth 0.08 interpolation within state |

---

## INVARIANTS ACTIVE (All Remaining Modules)

1. Rate-of-change over absolute thresholds — fault signatures as dX*/dt
2. Cluster-conditional normalisation — no global statistical parameters
3. Physics-weighted ML — loss functions reflect physical sensor hierarchy
4. Temporal pattern over absolute level — overloading = rising T*, not high T
5. Conservative baseline is safer production — false-negative on transition safer than false-positive shutdown
6. Sensor failure ≠ process failure — single-channel anomaly ≠ multi-channel fault
7. Climate-agnostic normalisation — cluster-relative ΔT*, not ambient-relative
8. Clean normal baseline is non-negotiable — threshold quality bounded by normal training purity (C-16)
9. Cross-window temporal context required — TCN L2 on z_t sequences only architecturally sound
10. Vanishing gradient ceiling on recurrent L2 — TCN superior for N_windows > 50
11. Dual time-scale architecture — CUSUM and rolling baseline in parallel, NEVER merged
12. Sequence length physically justified — uniform lengths = dimensionally incorrect
13. Synthetic-to-real gap = acknowledged bounded limitation — surfaced in every API response Field 7
14. Sensor health is sidecar diagnostic, NEVER prediction override — C-28 / Principle 14

**Architectural constants:**
- C-22: P_hydraulic = 55.2 kW (corrects Zenodo "10 kW" documentation error)
- C-25: Adaptive Threshold Paradox — L3/L4 NEVER merged
- C-26: Real-world expected F1 = 0.65–0.85 (model trained on synthetic data)
- C-28: M8p6 sensor sensitivity guardrail — sidecar addendum, never overrides
- q = 0.110058: M4 LSTM-AE threshold — PERMANENTLY LOCKED, never retrain
- H = 5.0: CUSUM alarm threshold
- Invariant 19: score_A → L4, score_B → L3 CUSUM, score_C → M7 XGBoost

---

## NEXT STEPS — IMMEDIATE

1. **M10 Phase 1:** Write `main.py` — FastAPI entry point, lifespan model loading, router registration, all 8 routes with placeholder responses. Confirm before executing.
2. **M10 Phase 2:** Full inference pipeline — 4-layer stack, 7-field output, M8p6 addendum, alert state machine.
3. **M10 Phase 3:** React JSX frontend integration.
4. **M12 (before M11):** Adversarial validation suite.
5. **M11:** Docker + HF Spaces deployment.

---

*PumpSmart v14.2 + M8p6 | Context Document v2.0 | Supersedes Sequential_Industry_grade_verification_pathway.md | May 2026*
