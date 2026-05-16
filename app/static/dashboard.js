// =============================================================================
// app/static/dashboard.js
// PumpSmart v14.2 — API polling layer
// Bridges the React JSX frontend to the FastAPI backend.
//
// Exposes window.PumpSmartAPI — consumed by pumpsmart_full_v2.jsx
// All fetch() calls go through this module; JSX never calls fetch() directly.
//
// Polling schedule:
//   /health              → every 30s (HEALTH_POLL_MS)
//   /api/anomaly_detect  → every ~50s (INFERENCE_POLL_MS)
//   /api/physics_context → on-demand (per prediction, cached)
// =============================================================================

(function () {
  'use strict';

  const BASE = window.PUMPSMART_API_BASE || '';
  const PUMP_ID = window.PUMPSMART_PUMP_ID || 'PUMP-0032';

  // ── Internal state ────────────────────────────────────────────────────────
  let _listeners = {};          // event → [callbacks]
  let _healthTimer = null;
  let _inferenceTimer = null;
  let _latestPrediction = null;
  let _latestHealth = null;
  let _physicsCache = {};       // label_int → physics context (cached)
  let _sensorConnected = {      // mirrors Day-1 config — updated by Sensor Plugin
    mot_sv: true, pmp_sv: true, mot_tv: true, pmp_pv: true,
    temp_sv: true, pres_sv: true, pmp_tv: true, mot_pv: true,
  };
  let _currentCluster = 'steady_state';

  // ── Event bus (minimal pub/sub for React ↔ polling) ───────────────────────
  function on(event, cb) {
    if (!_listeners[event]) _listeners[event] = [];
    _listeners[event].push(cb);
  }
  function off(event, cb) {
    if (_listeners[event])
      _listeners[event] = _listeners[event].filter(fn => fn !== cb);
  }
  function emit(event, data) {
    (_listeners[event] || []).forEach(cb => { try { cb(data); } catch(e) {} });
  }

  // ── Fetch helpers ─────────────────────────────────────────────────────────
  async function _get(path) {
    const r = await fetch(BASE + path);
    if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
    return r.json();
  }

  async function _post(path, body) {
    const r = await fetch(BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const txt = await r.text();
      throw new Error(`POST ${path} → ${r.status}: ${txt}`);
    }
    return r.json();
  }

  // ── /health polling ───────────────────────────────────────────────────────
  async function pollHealth() {
    try {
      const data = await _get('/health');
      _latestHealth = data;
      emit('health', data);
    } catch (e) {
      emit('health_error', { error: e.message });
    }
  }

  function startHealthPolling() {
    pollHealth();
    _healthTimer = setInterval(pollHealth, window.HEALTH_POLL_MS || 30000);
  }

  // ── /api/anomaly_detect polling ───────────────────────────────────────────
  // In real deployment: SCADA pushes windows via POST.
  // In demo/shadow mode: frontend generates synthetic windows for UI testing.
  async function pollInference() {
    // Only run if enough sensors are connected
    const connectedCount = Object.values(_sensorConnected).filter(Boolean).length;
    if (connectedCount < 4) {
      emit('inference_paused', { reason: 'insufficient_sensors', connected: connectedCount });
      return;
    }

    try {
      const window50x8 = _generateDemoWindow(_currentCluster);
      const payload = {
        window: window50x8,
        pump_id: PUMP_ID,
        cluster: _currentCluster,
        timestamp_utc: new Date().toISOString(),
      };
      const data = await _post('/api/anomaly_detect', payload);
      _latestPrediction = data;

      // Cache physics context for this label (if not already cached)
      if (data.fault_label !== undefined) {
        const labelInt = _labelNameToInt(data.fault_label);
        if (labelInt !== null && !_physicsCache[labelInt]) {
          _fetchPhysicsContext(labelInt);
        }
      }

      emit('prediction', data);
    } catch (e) {
      emit('inference_error', { error: e.message });
    }
  }

  function startInferencePolling() {
    // Stagger 5s after health poll to avoid startup burst
    setTimeout(() => {
      pollInference();
      _inferenceTimer = setInterval(pollInference, window.INFERENCE_POLL_MS || 50000);
    }, 5000);
  }

  // ── /api/physics_context ──────────────────────────────────────────────────
  async function _fetchPhysicsContext(labelInt) {
    try {
      const data = await _get(`/api/physics_context?label=${labelInt}`);
      _physicsCache[labelInt] = data;
      emit('physics_context', { label: labelInt, data });
    } catch (e) { /* silently cache miss */ }
  }

  function getPhysicsContext(labelInt) {
    return _physicsCache[labelInt] || null;
  }

  // ── /api/acknowledge ──────────────────────────────────────────────────────
  // v5.0-A: operational reset ONLY — does NOT write active-learning row
  async function acknowledge(actionTaken, operatorId = '') {
    try {
      const data = await _post('/api/acknowledge', {
        pump_id: PUMP_ID,
        action_taken: actionTaken,
        operator_id: operatorId,
        timestamp_utc: new Date().toISOString(),
      });
      emit('acknowledged', data);
      return data;
    } catch (e) {
      emit('acknowledge_error', { error: e.message });
      throw e;
    }
  }

  // ── /api/operator_verdict ─────────────────────────────────────────────────
  // v5.0-A: ONLY active-learning write point (Predictions tab)
  async function submitVerdict(predictionId, verdict, opts = {}) {
    try {
      const data = await _post('/api/operator_verdict', {
        prediction_id: predictionId,
        verdict,                                    // CORRECT | INCORRECT | UNSURE
        operator_correct_label: opts.correctLabel || null,
        physical_inspection_done: opts.inspectionDone || false,
        inspection_notes: opts.notes || '',
        operator_id: opts.operatorId || '',
        consent_granted_by: opts.operatorId || '',
      });
      emit('verdict_submitted', data);
      return data;
    } catch (e) {
      emit('verdict_error', { error: e.message });
      throw e;
    }
  }

  // ── /api/validate_model ───────────────────────────────────────────────────
  async function validateModel() {
    return _get('/api/validate_model');
  }

  // ── /api/select_pump (M9 physics) ────────────────────────────────────────
  async function selectPump(spec) {
    return _post('/api/select_pump', spec);
  }

  // ── /api/household ────────────────────────────────────────────────────────
  async function householdAdvisor(params) {
    const qs = new URLSearchParams(params).toString();
    return _get(`/api/household?${qs}`);
  }

  // ── Sensor state management ───────────────────────────────────────────────
  function setSensorConnected(sensorId, connected) {
    _sensorConnected[sensorId] = connected;
    const count = Object.values(_sensorConnected).filter(Boolean).length;
    emit('sensor_state_changed', { sensorId, connected, total_connected: count });
  }

  function getSensorState() {
    return { ..._sensorConnected };
  }

  function setCluster(cluster) {
    _currentCluster = cluster;
  }

  // ── Stop polling ──────────────────────────────────────────────────────────
  function stopPolling() {
    if (_healthTimer)    clearInterval(_healthTimer);
    if (_inferenceTimer) clearInterval(_inferenceTimer);
  }

  // ── Accessors ─────────────────────────────────────────────────────────────
  function getLatestPrediction() { return _latestPrediction; }
  function getLatestHealth()     { return _latestHealth; }

  // ── Demo window generator ─────────────────────────────────────────────────
  // Generates synthetic M3-normalised 50×8 windows for shadow/demo mode.
  // In real deployment this is replaced by SCADA data ingestion.
  // Cluster-conditional ranges match M2 K-Means centroids.
  const CLUSTER_BASELINES = {
    startup:      [0.85, 0.80, 0.75, 0.20, 0.72, 0.60, 0.65, 0.55],
    steady_state: [0.45, 0.42, 0.50, 0.95, 0.55, 0.80, 0.52, 0.90],
    high_load:    [0.55, 0.52, 0.65, 0.98, 0.68, 0.72, 0.62, 0.98],
    cooldown:     [0.30, 0.28, 0.40, 0.45, 0.38, 0.35, 0.28, 0.35],
  };

  function _generateDemoWindow(cluster) {
    const baseline = CLUSTER_BASELINES[cluster] || CLUSTER_BASELINES.steady_state;
    const window = [];
    for (let t = 0; t < 50; t++) {
      const row = baseline.map((b, ch) => {
        const noise = (Math.random() - 0.5) * 0.04;
        const drift = 0;   // normal operation — no drift
        return Math.max(0, Math.min(3, b + noise + drift));
      });
      window.push(row);
    }
    return window;
  }

  // ── Label name → int lookup ───────────────────────────────────────────────
  const LABEL_MAP_INV = {
    'normal': 0, 'bearing_wear': 1, 'impeller_imbalance': 2,
    'cavitation': 3, 'seal_failure': 4, 'overloading': 5, 'sensor_failure': 6,
    'bearing_overloading': 7, 'cavitation_seal': 8, 'imbalance_bearing': 9,
    'seal_cavitation_high_head': 10, 'overloading_bearing': 11, 'imbalance_cavitation': 12,
    'bearing_mot_sv_masked': 13, 'cavitation_pres_sv_masked': 14,
    'seal_pres_sv_drifting': 15, 'overloading_temp_sv_stuck': 16,
    'imbalance_pmp_sv_flatline': 17, 'cavitation_intermittent': 18,
    'seal_failure_fast': 19, 'overloading_cyclic': 20, 'bearing_wear_gradual': 21,
  };
  function _labelNameToInt(name) {
    return LABEL_MAP_INV[name] !== undefined ? LABEL_MAP_INV[name] : null;
  }

  // ── Initialise ────────────────────────────────────────────────────────────
  function init() {
    startHealthPolling();
    startInferencePolling();
  }

  // ── Public API ────────────────────────────────────────────────────────────
  window.PumpSmartAPI = {
    // Lifecycle
    init,
    stopPolling,
    // Events
    on,
    off,
    // Endpoints
    acknowledge,
    submitVerdict,
    validateModel,
    selectPump,
    householdAdvisor,
    getPhysicsContext,
    // State
    getLatestPrediction,
    getLatestHealth,
    setSensorConnected,
    getSensorState,
    setCluster,
    // Utilities
    generateDemoWindow: _generateDemoWindow,
  };

})();
