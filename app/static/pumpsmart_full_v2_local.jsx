// =============================================================================
// PumpSmart v14.2 — Industrial Dashboard (PRODUCTION) — v5.1
// pumpsmart_full_v2_local.jsx
//
// v5.1 — M10 Phase 2.5 integration:
//   - Server-side sensor history (multi-client safe — 5 testers share timeline)
//   - Live chart on Dashboard polls /api/sensor_history every 1s (last 5 min)
//   - Analytics tab polls every 5s (last 1 hour)
//   - History tab on-demand load via button (last 1h / 6h / 24h)
//   - DS-C adaptive downsampling (last 5 min full-res + LTTB for older)
//   - History preserved across acknowledge (forensic retention)
//
// Data sources:
//   /health → polled every 30s regardless of sensor state (server heartbeat)
//   /api/anomaly_detect → polled ONLY when at least 1 sensor is marked active
//   /api/sensor_history → tab-driven polling
//
// Sensor connection gate:
//   connectedCount === 0 → Dashboard/Analytics/Predictions/History show gate
//   connectedCount > 0   → inference polling starts, data flows
//
// For M12 local testing:
//   1. Mark sensors as active in Sensor Plugin
//   2. Run: python src/module_12b_adversarial_runner.py --mode smoke
//   3. Dashboard receives live predictions + history chart updates
// =============================================================================

const ALERT_COLORS = { NORMAL:"#00e676", WATCH:"#ffcc00", WARN:"#ff8800", DANGER:"#ff2244" };

const SENSOR_CHANNELS = [
  { id:"mot_sv",  name:"Motor Vibration (RMS)",      unit:"mm/s", ch:"Mot.SV",  icon:"〰️", idx:0, color:"#00d4ff",
    desc:"Motor bearing side broadband peak acceleration envelope (SCADA composite — not ISO 10816-3 RMS)" },
  { id:"pmp_sv",  name:"Pump Vibration (RMS)",        unit:"mm/s", ch:"Pmp.SV",  icon:"〰️", idx:1, color:"#ffcc00",
    desc:"Pump casing broadband peak acceleration envelope" },
  { id:"mot_tv",  name:"Motor Winding Temperature",   unit:"°C",   ch:"Mot.TV",  icon:"🌡️", idx:2, color:"#ff8800",
    desc:"Stator winding temperature — thermal overload indicator" },
  { id:"pmp_pv",  name:"Pump Discharge Pressure",     unit:"bar",  ch:"Pmp.PV",  icon:"🔵", idx:3, color:"#9b7fe8",
    desc:"Main discharge pressure — primary operating parameter" },
  { id:"temp_sv", name:"Bearing Housing Temperature", unit:"°C",   ch:"Temp.SV", icon:"🌡️", idx:4, color:"#ff4466",
    desc:"Bearing housing temperature — early fault indicator" },
  { id:"pres_sv", name:"Suction Side Pressure",       unit:"bar",  ch:"Pres.SV", icon:"🔵", idx:5, color:"#00e676",
    desc:"Suction-side pressure — NPSHa monitoring" },
  { id:"pmp_tv",  name:"Pump Casing Temperature",     unit:"°C",   ch:"Pmp.TV",  icon:"🌡️", idx:6, color:"#e67e22",
    desc:"Pump body temperature — flash evaporation indicator on shutdown" },
  { id:"mot_pv",  name:"Motor Power Draw",            unit:"kW",   ch:"Mot.PV",  icon:"⚡", idx:7, color:"#00b4d8",
    desc:"Electrical power input — overloading detection channel" },
];

const CLUSTER_LABELS = ["Startup","Steady-state","High-load","Cooldown"];
const CLUSTER_COLORS = ["#ffbb00","#00e676","#ff8800","#00b4d8"];
const CLUSTER_IDS    = ["startup","steady_state","high_load","cooldown"];

const MODEL_DISCLAIMER =
  "Trained on CIRA-anchored physics-synthetic data for 110 kW 7-stage pump at " +
  "2980 RPM, 40 bar, 45 m³/h. Advisory only — verify physically. " +
  "Real-world F1 expected 0.65–0.85 (C-26). Single-pump v14.2.";

const ONBOARDING_POINTS = [
  { icon:"⚙️", title:"Exact pump specification required",
    text:"Trained exclusively for 110 kW, 7-stage centrifugal pump at 2980 RPM, 40 bar, 45 m³/h." },
  { icon:"🧠", title:"Advisory only — not autonomous control",
    text:"All predictions require human review before any maintenance action." },
  { icon:"🔬", title:"Trained on physics-synthetic data",
    text:"Real-world F1 expected 0.65–0.85 (C-26). Always verify predictions physically." },
  { icon:"👷", title:"Certified engineer verification mandatory",
    text:"Every WARN or DANGER prediction must be inspected by a qualified engineer." },
  { icon:"📊", title:"Confidence below 70% = UNKNOWN FAULT",
    text:"Low confidence does not mean no fault — it means multiple fault types are plausible. Inspect regardless." },
  { icon:"🔌", title:"Sensor data quality is your responsibility",
    text:"Calibration, signal conditioning and wiring quality directly impact prediction accuracy." },
];

// =============================================================================
// UTILITY COMPONENTS
// =============================================================================
function EmptyState({ icon="📡", message, sub, action }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center",
                  justifyContent:"center", padding:"48px 16px", gap:12 }}>
      <div style={{ fontSize:32, opacity:.35 }}>{icon}</div>
      <div style={{ fontSize:14, color:"#3a6a7a", textAlign:"center", fontWeight:500 }}>{message}</div>
      {sub && <div style={{ fontSize:12, color:"#1a4050", textAlign:"center", lineHeight:1.6 }}>{sub}</div>}
      {action}
    </div>
  );
}

function SensorGate({ onGoToSensor }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center",
                  justifyContent:"center", flex:1, padding:48, gap:16 }}>
      <div style={{ fontSize:48, opacity:.4 }}>🔌</div>
      <div style={{ fontSize:18, fontWeight:700, color:"#e0f0ff", textAlign:"center" }}>
        No sensor data source active
      </div>
      <div style={{ fontSize:13, color:"#3a6a7a", textAlign:"center", lineHeight:1.8, maxWidth:480 }}>
        This tab shows live data from the inference pipeline.<br/>
        No predictions can be made until a data source is active.
      </div>
      <div style={{ padding:"12px 18px", background:"rgba(0,180,220,.07)",
        border:"1px solid rgba(0,180,220,.2)", borderRadius:10,
        fontSize:12, color:"#6ab0cc", lineHeight:1.8, maxWidth:520 }}>
        <div style={{ fontWeight:600, color:"#00d4ff", marginBottom:6 }}>How to start receiving data:</div>
        <div>
          <strong style={{ color:"#c0d8f0" }}>M12 local test:</strong> Run{" "}
          <code style={{ background:"rgba(0,180,220,.12)", padding:"2px 7px", borderRadius:4, fontSize:11 }}>
            python src/module_12b_adversarial_runner.py
          </code>
          <br/>
          <strong style={{ color:"#c0d8f0" }}>SCADA/PLC:</strong> POST normalised 50×8 windows to{" "}
          <code style={{ background:"rgba(0,180,220,.12)", padding:"2px 7px", borderRadius:4, fontSize:11 }}>
            POST /api/anomaly_detect
          </code>
        </div>
        <div style={{ marginTop:8, color:"#3a7a8a" }}>
          After starting data flow, go to <strong style={{ color:"#c0d8f0" }}>Sensor Plugin</strong> and
          mark the channels that are actively sending data as "Data active".
          The inference pipeline will start automatically.
        </div>
      </div>
      <button onClick={onGoToSensor}
        style={{ padding:"10px 24px", borderRadius:9, background:"rgba(0,180,220,.15)",
          border:"1px solid rgba(0,180,220,.35)", color:"#00d4ff",
          cursor:"pointer", fontSize:13, fontWeight:600 }}>
        🔌 Go to Sensor Plugin →
      </button>
    </div>
  );
}

function Card({ children, style={}, animate=true }) {
  const [vis, setVis] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVis(true), 50); return () => clearTimeout(t); }, []);
  return (
    <div style={{
      background:"rgba(6,18,36,0.88)", border:"1px solid rgba(20,60,100,.5)",
      borderRadius:12, padding:16, backdropFilter:"blur(8px)",
      transition:"opacity .4s ease, transform .4s ease",
      opacity:animate?(vis?1:0):1, transform:animate?(vis?"translateY(0)":"translateY(14px)"):"none",
      ...style
    }}>{children}</div>
  );
}

function CardHdr({ title, right }) {
  return (
    <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:14 }}>
      <span style={{ fontSize:11, color:"#5a9ab0", fontWeight:500, letterSpacing:.8, textTransform:"uppercase" }}>{title}</span>
      {right && <div>{right}</div>}
    </div>
  );
}

// =============================================================================
// SENSOR HISTORY CHART — Chart.js wrapper for server-side history payload
// Used on Dashboard (live, 5 min), Analytics (1 hr), History (on-demand 24 hr)
// =============================================================================
function SensorHistoryChart({ historyData, height=220, channelsToShow=null, showDerived=false }) {
  const canv = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!historyData || historyData.n_points === 0 || !canv.current) {
      if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }
      return;
    }
    if (!window.Chart) return;

    const labels = historyData.timestamps.map((ts, i) =>
      i % Math.max(1, Math.floor(historyData.n_points / 10)) === 0
        ? new Date(ts).toLocaleTimeString()
        : "");

    const channels = channelsToShow ||
      SENSOR_CHANNELS.filter((_,i) => i < 4).map(s => s.ch);   // first 4 default

    const datasets = channels.map(chName => {
      const meta = SENSOR_CHANNELS.find(s => s.ch === chName);
      return {
        label: meta ? meta.ch : chName,
        data: historyData.channels[chName] || [],
        borderColor: meta ? meta.color : "#888",
        backgroundColor: "transparent",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
      };
    });

    if (showDerived) {
      datasets.push({
        label: "score_A",
        data: historyData.score_A,
        borderColor: "#ff4466", borderWidth: 1.5, borderDash:[4,4],
        backgroundColor: "transparent", pointRadius: 0, yAxisID: "y1",
      });
      datasets.push({
        label: "CUSUM S_n",
        data: historyData.cusum_Sn,
        borderColor: "#ffcc00", borderWidth: 1.5,
        backgroundColor: "transparent", pointRadius: 0, yAxisID: "y1",
      });
    }

    const config = {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: "#8ab0c8", font: { size: 10 }, boxWidth: 14 }, position: "bottom" },
          tooltip: { backgroundColor: "rgba(4,12,26,.95)", titleColor: "#00d4ff",
                     bodyColor: "#c0d8f0", borderColor: "rgba(20,60,100,.6)", borderWidth: 1 },
        },
        scales: {
          x: { ticks: { color: "#3a6a7a", font: { size: 9 }, maxRotation: 0, autoSkip: true },
               grid: { color: "rgba(20,55,95,.2)" } },
          y: { ticks: { color: "#3a6a7a", font: { size: 9 } },
               grid: { color: "rgba(20,55,95,.2)" },
               title: { display: true, text: "Normalised", color: "#3a6a7a", font: { size: 10 } } },
          ...(showDerived ? { y1: { position: "right",
                                     ticks: { color: "#ffcc00", font: { size: 9 } },
                                     grid: { display: false },
                                     title: { display: true, text: "score_A / S_n", color: "#ffcc00", font: { size: 10 } } } } : {}),
        },
      },
    };

    if (chartRef.current) {
      chartRef.current.data = config.data;
      chartRef.current.options = config.options;
      chartRef.current.update("none");
    } else {
      chartRef.current = new window.Chart(canv.current.getContext("2d"), config);
    }
  }, [historyData, channelsToShow, showDerived]);

  useEffect(() => () => { if (chartRef.current) chartRef.current.destroy(); }, []);

  if (!historyData || historyData.n_points === 0) {
    return (
      <div style={{ height, display:"flex", alignItems:"center", justifyContent:"center",
                    color:"#2a5060", fontSize:12 }}>
        ⏳ Waiting for sensor data to accumulate in server-side history buffer…
      </div>
    );
  }

  return (
    <div style={{ position: "relative", height }}>
      <canvas ref={canv}/>
      <div style={{ position:"absolute", top:4, right:8, fontSize:9, color:"#2a5060" }}>
        {historyData.n_points} pts / {historyData.original_n} raw · {historyData.downsample_method}
      </div>
    </div>
  );
}

// =============================================================================
// RISK GAUGE
// =============================================================================
function RiskGauge({ alertState }) {
  const canv = useRef();
  const prevState = useRef(alertState);
  const animRef = useRef();
  const TARGETS = { NORMAL:0.05, WATCH:0.38, WARN:0.63, DANGER:0.88 };
  const color = ALERT_COLORS[alertState] || "#00e676";
  const label = alertState || "NORMAL";

  useEffect(() => {
    const target = TARGETS[alertState] || 0.05;
    const stateChanged = prevState.current !== alertState;
    let cur = stateChanged ? target : (TARGETS[prevState.current] || 0.05);
    prevState.current = alertState;

    const draw = (n) => {
      const c = canv.current; if (!c) return;
      const ctx = c.getContext("2d"); const cx=120, cy=150, R=105;
      ctx.clearRect(0,0,240,160);
      const bg = ctx.createRadialGradient(cx,cy,50,cx,cy,R+25);
      bg.addColorStop(0,"rgba(8,24,52,.9)"); bg.addColorStop(1,"rgba(4,12,28,.1)");
      ctx.beginPath(); ctx.arc(cx,cy,R+20,-Math.PI,0); ctx.fillStyle=bg; ctx.fill();
      [{ c:"#00e676",s:Math.PI,e:Math.PI*1.28 },{ c:"#aacc00",s:Math.PI*1.28,e:Math.PI*1.54 },
       { c:"#ffbb00",s:Math.PI*1.54,e:Math.PI*1.76 },{ c:"#ff6600",s:Math.PI*1.76,e:Math.PI*1.92 },
       { c:"#ff2244",s:Math.PI*1.92,e:Math.PI*2 }].forEach(({c:col,s,e}) => {
        ctx.beginPath(); ctx.arc(cx,cy,R,s,e); ctx.strokeStyle=col+"44"; ctx.lineWidth=24; ctx.lineCap="butt"; ctx.stroke();
        ctx.beginPath(); ctx.arc(cx,cy,R,s,e); ctx.strokeStyle=col; ctx.lineWidth=14; ctx.stroke();
      });
      const na = Math.PI+n*Math.PI, nx=cx+Math.cos(na)*(R-18), ny=cy+Math.sin(na)*(R-18);
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(nx,ny); ctx.strokeStyle=color+"55"; ctx.lineWidth=10; ctx.lineCap="round"; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(nx,ny); ctx.strokeStyle=color; ctx.lineWidth=3; ctx.lineCap="round"; ctx.stroke();
      ctx.beginPath(); ctx.arc(nx,ny,5,0,Math.PI*2); ctx.fillStyle=color; ctx.shadowColor=color; ctx.shadowBlur=14; ctx.fill(); ctx.shadowBlur=0;
      ctx.beginPath(); ctx.arc(cx,cy,11,0,Math.PI*2); ctx.fillStyle="#060f1c"; ctx.fill();
      ctx.beginPath(); ctx.arc(cx,cy,5,0,Math.PI*2); ctx.fillStyle=color; ctx.fill();
      ctx.font="bold 18px Inter,sans-serif"; ctx.fillStyle=color; ctx.textAlign="center";
      ctx.shadowColor=color; ctx.shadowBlur=16; ctx.fillText(label,cx,cy-30); ctx.shadowBlur=0;
      ctx.font="10px Inter,sans-serif"; ctx.fillStyle="rgba(140,180,210,.65)"; ctx.fillText("Failure Risk Level",cx,cy-14);
    };

    cancelAnimationFrame(animRef.current);
    if (stateChanged) { draw(target); }
    else {
      const step = () => { cur+=(target-cur)*.08; draw(cur); if(Math.abs(cur-target)>.005) animRef.current=requestAnimationFrame(step); else draw(target); };
      step();
    }
    return () => cancelAnimationFrame(animRef.current);
  }, [alertState, color, label]);

  return <canvas ref={canv} width={240} height={160} style={{ display:"block", margin:"0 auto" }}/>;
}

// =============================================================================
// ONBOARDING POPUP
// =============================================================================
function OnboardingPopup({ onAccept, onViewGuide }) {
  const [checked, setChecked] = useState(false);
  return (
    <div style={{ position:"fixed", inset:0, background:"rgba(2,6,14,.92)", backdropFilter:"blur(16px)",
                  display:"flex", alignItems:"center", justifyContent:"center", zIndex:200 }}>
      <div style={{ background:"linear-gradient(160deg,rgba(8,18,36,.98),rgba(4,10,22,.99))",
                    border:"1px solid rgba(0,180,220,.25)", borderRadius:16, padding:32,
                    maxWidth:620, width:"92%", maxHeight:"88vh", display:"flex",
                    flexDirection:"column", overflow:"hidden", boxShadow:"0 32px 80px rgba(0,0,0,.7)" }}>
        <div style={{ display:"flex", alignItems:"center", gap:14, marginBottom:20 }}>
          <div style={{ fontSize:36 }}>⚙️</div>
          <div>
            <div style={{ fontSize:18, fontWeight:700, color:"#e0f0ff" }}>Before you proceed</div>
            <div style={{ fontSize:12, color:"#3a6a7a", marginTop:3 }}>PumpSmart v14.2 — applicability acknowledgement</div>
          </div>
        </div>
        <div style={{ background:"rgba(0,180,220,.07)", border:"1px solid rgba(0,180,220,.2)",
                      borderRadius:10, padding:"12px 16px", marginBottom:16 }}>
          <div style={{ fontSize:11, color:"#3a6a7a", textTransform:"uppercase", letterSpacing:.8, marginBottom:8 }}>
            Designed exclusively for
          </div>
          <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:10 }}>
            {["⚡ 110 kW","🔧 7-stage centrifugal","📊 40 bar","🌀 2980 RPM","💧 45 m³/h","📐 IEC 315"].map((s,i) => (
              <span key={i} style={{ fontSize:11, padding:"3px 10px", borderRadius:6,
                background:"rgba(0,200,220,.1)", border:"1px solid rgba(0,200,220,.25)",
                color:"#00d4ff", fontWeight:600 }}>{s}</span>
            ))}
          </div>
          <div style={{ fontSize:12, color:"#ff8800" }}>⚠️ Any other pump specification will produce unreliable predictions.</div>
        </div>
        <div style={{ overflowY:"auto", flex:1, marginBottom:14 }}>
          {ONBOARDING_POINTS.map((pt,i) => (
            <div key={i} style={{ display:"flex", gap:12, padding:"10px 14px", marginBottom:8,
              background:"rgba(4,12,26,.6)", border:"1px solid rgba(20,55,95,.4)", borderRadius:10 }}>
              <span style={{ fontSize:20 }}>{pt.icon}</span>
              <div>
                <div style={{ fontSize:12, fontWeight:600, color:"#c0d8f0", marginBottom:3 }}>{pt.title}</div>
                <div style={{ fontSize:12, color:"#5a8aaa", lineHeight:1.5 }}>{pt.text}</div>
              </div>
            </div>
          ))}
        </div>
        <div style={{ borderTop:"1px solid rgba(20,55,95,.5)", paddingTop:14 }}>
          <label style={{ display:"flex", alignItems:"flex-start", gap:12, cursor:"pointer", marginBottom:12 }}>
            <div onClick={() => setChecked(c => !c)}
              style={{ width:22, height:22, borderRadius:6, flexShrink:0, marginTop:1, cursor:"pointer",
                border:`2px solid ${checked?"#00d4ff":"rgba(40,90,130,.6)"}`,
                background:checked?"rgba(0,200,220,.15)":"transparent",
                display:"flex", alignItems:"center", justifyContent:"center" }}>
              {checked && <span style={{ color:"#00d4ff", fontSize:14, fontWeight:700 }}>✓</span>}
            </div>
            <span style={{ fontSize:13, color:"#8ab0c8", lineHeight:1.6 }}>
              I confirm my pump matches all specifications above. I understand all predictions are{" "}
              <strong style={{ color:"#c0d8f0" }}>advisory only</strong> and must be verified by a qualified engineer.
            </span>
          </label>
          <div style={{ display:"flex", gap:10 }}>
            <button onClick={onViewGuide} style={{ flex:1, padding:10, borderRadius:9, cursor:"pointer",
              background:"rgba(20,55,95,.3)", border:"1px solid rgba(40,90,130,.4)", color:"#5a9ab0", fontSize:12 }}>
              📖 Read guide first
            </button>
            <button onClick={() => checked && onAccept()} style={{ flex:2, padding:11, borderRadius:9,
              cursor:checked?"pointer":"not-allowed", fontSize:14, fontWeight:700,
              border:`1px solid ${checked?"rgba(0,200,220,.4)":"rgba(30,70,110,.3)"}`,
              background:checked?"rgba(0,180,220,.18)":"rgba(20,55,95,.2)",
              color:checked?"#00d4ff":"#2a5060" }}>
              {checked ? "✓ Acknowledged — enter dashboard" : "Tick the box above to continue"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// ALERT POPUP
// =============================================================================
function AlertPopup({ prediction, alertState, onAcknowledge, onDismiss }) {
  if (!prediction) return null;
  const color = ALERT_COLORS[alertState] || "#ff8800";
  return (
    <div style={{ position:"absolute", inset:0, background:"rgba(2,8,18,.75)",
                  display:"flex", alignItems:"center", justifyContent:"center", zIndex:50 }}>
      <div style={{ background:"#060f1e", border:`1px solid ${color}55`, borderRadius:14,
                    padding:24, maxWidth:500, width:"90%", boxShadow:`0 0 50px ${color}18` }}>
        <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:16 }}>
          <div style={{ width:44, height:44, borderRadius:10, background:`${color}18`,
            border:`1px solid ${color}44`, display:"flex", alignItems:"center",
            justifyContent:"center", fontSize:22 }}>
            {alertState==="DANGER"?"🔴":"⚠️"}
          </div>
          <div>
            <div style={{ fontSize:16, fontWeight:700, color }}>{alertState} — {prediction.fault_label}</div>
            <div style={{ fontSize:11, color:"#3a6a7a", marginTop:2 }}>
              Confidence {prediction.confidence_pct?.toFixed(1)}% · {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
        {[
          { k:"🔬 Physical condition",  v:prediction.probable_physical_condition },
          { k:"⏳ Risk if ignored",     v:prediction.operational_risk_if_ignored, c:"#ff8800" },
          { k:"🔧 Recommended action", v:prediction.recommended_action },
        ].map((row,i) => (
          <div key={i} style={{ display:"grid", gridTemplateColumns:"140px 1fr", gap:10,
            paddingBottom:8, borderBottom:"1px solid rgba(20,55,95,.3)", marginBottom:8 }}>
            <div style={{ fontSize:11, color:"#3a6a7a", fontWeight:500 }}>{row.k}</div>
            <div style={{ fontSize:12, color:row.c||"#c0d8f0", lineHeight:1.5 }}>{row.v||"—"}</div>
          </div>
        ))}
        <div style={{ marginBottom:12, padding:"8px 12px", background:"rgba(0,150,210,.06)",
          border:"1px solid rgba(0,150,210,.2)", borderRadius:8, fontSize:11, color:"#4a8aaa", lineHeight:1.6 }}>
          ℹ️ <strong style={{ color:"#6ab0cc" }}>Acknowledge</strong> = operational reset (clears CUSUM, z_t buffer, rolling baseline).
          Sensor history is preserved for forensic review.
          To rate model accuracy, use the <strong style={{ color:"#6ab0cc" }}>Predictions tab</strong> after physical inspection.
        </div>
        <div style={{ display:"flex", gap:10 }}>
          <button onClick={onAcknowledge} style={{ flex:2, padding:10, borderRadius:8, cursor:"pointer",
            background:"rgba(0,100,180,.2)", border:"1px solid rgba(0,140,220,.4)",
            color:"#00b4d8", fontSize:13, fontWeight:600 }}>
            ✅ Acknowledge — reset alarm
          </button>
          <button onClick={onDismiss} style={{ flex:1, padding:10, borderRadius:8, cursor:"pointer",
            background:"rgba(30,8,14,.5)", border:"1px solid rgba(150,20,40,.3)",
            color:"#ff4466", fontSize:13 }}>
            ❌ Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN APP
// =============================================================================
function App() {
  const [tab, setTab] = useState("dashboard");
  const [onboardingDone, setOnboardingDone] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [activeCluster, setActiveCluster] = useState(1);
  const [verdictSent, setVerdictSent] = useState(null);

  const [sensorActive, setSensorActive] = useState({
    mot_sv:false, pmp_sv:false, mot_tv:false, pmp_pv:false,
    temp_sv:false, pres_sv:false, pmp_tv:false, mot_pv:false,
  });

  const [healthData, setHealthData]     = useState(null);
  const [prediction, setPrediction]     = useState(null);
  const [alertState, setAlertState]     = useState("NORMAL");
  const [eventLog, setEventLog]         = useState([]);
  const [showPopup, setShowPopup]       = useState(false);
  const [apiError, setApiError]         = useState(null);
  const [lastUpdate, setLastUpdate]     = useState(null);
  const [pollingActive, setPollingActive] = useState(false);

  // ── Phase 2.5: server-side history state (multi-client safe) ──
  const [liveChartData,    setLiveChartData]    = useState(null);   // dashboard (1s)
  const [analyticsData,    setAnalyticsData]    = useState(null);   // analytics (5s)
  const [historyTabData,   setHistoryTabData]   = useState(null);   // history (on-demand)
  const [historyTabRange,  setHistoryTabRange]  = useState(24);     // hours
  const [historyTabLoading,setHistoryTabLoading]= useState(false);
  const [historyBufferState, setHistoryBufferState] = useState(null);

  // Channel visibility toggle for charts
  const [chartChannels, setChartChannels] = useState(
    SENSOR_CHANNELS.filter((_,i) => i < 4).map(s => s.ch)
  );

  const connectedCount = Object.values(sensorActive).filter(Boolean).length;
  const noSensors      = connectedCount === 0;

  const GATED_TABS = ["dashboard","sensor","analytics","predictions","history","settings"];
  const DATA_TABS  = ["dashboard","analytics","predictions","history"];

  // ── API listeners — set up when onboarding done ───────────────────────────
  useEffect(() => {
    if (!onboardingDone) return;

    window.PumpSmartAPI.on("health", (data) => {
      setHealthData(data);
      setApiError(null);
      setLastUpdate(new Date().toLocaleTimeString());
    });

    window.PumpSmartAPI.on("health_error", () => {
      setApiError("Cannot reach server. Check uvicorn is running.");
    });

    window.PumpSmartAPI.on("prediction", (data) => {
      setPrediction(data);
      setAlertState(data.alert_state || "NORMAL");
      setLastUpdate(new Date().toLocaleTimeString());
      setApiError(null);
      setEventLog(prev => [{
        time:          new Date().toLocaleTimeString(),
        date:          new Date().toLocaleDateString(),
        state:         data.alert_state,
        fault:         data.fault_label,
        conf:          data.confidence_pct?.toFixed(1) + "%",
        score_A:       data.score_A?.toFixed(4),
        cusum_Sn:      data.cusum_Sn?.toFixed(4),
        prediction_id: data.prediction_id,
      }, ...prev].slice(0, 500));
      if (data.alert_state === "WARN" || data.alert_state === "DANGER") {
        setShowPopup(true);
      }
    });

    window.PumpSmartAPI.on("inference_paused", () => {});
    window.PumpSmartAPI.on("inference_error", (err) => {
      setApiError("Inference error: " + err.error);
    });
    window.PumpSmartAPI.on("acknowledged", () => {
      setShowPopup(false);
      setAlertState("NORMAL");
      // Note: server preserves sensor history — chart continues showing
      // the pre-acknowledge pattern for forensic review (v5.1)
    });

    // ── History event listener (Phase 2.5) ──
    const onHistoryUpdate = ({ tab: updateTab, data }) => {
      if (updateTab === "dashboard")  setLiveChartData(data);
      if (updateTab === "analytics")  setAnalyticsData(data);
      if (updateTab === "history")    setHistoryTabData(data);
    };
    const onHistoryError = (err) => {
      // non-fatal — history just won't update this tick
      console.warn("history fetch error:", err.error);
    };
    window.PumpSmartAPI.on("history_update", onHistoryUpdate);
    window.PumpSmartAPI.on("history_error",  onHistoryError);

    setPollingActive(true);
    window.PumpSmartAPI.init();

    return () => {
      window.PumpSmartAPI.off("history_update", onHistoryUpdate);
      window.PumpSmartAPI.off("history_error",  onHistoryError);
      window.PumpSmartAPI.stopPolling();
    };
  }, [onboardingDone]);

  // ── Tab-driven history polling (Phase 2.5) ──
  useEffect(() => {
    if (!onboardingDone || noSensors) {
      window.PumpSmartAPI?.stopHistoryPolling?.();
      return;
    }
    if (tab === "dashboard" || tab === "analytics") {
      window.PumpSmartAPI?.startHistoryPolling?.(tab);
    } else {
      window.PumpSmartAPI?.stopHistoryPolling?.();
    }
    return () => window.PumpSmartAPI?.stopHistoryPolling?.();
  }, [tab, onboardingDone, noSensors]);

  // ── When sensor count changes, update PumpSmartAPI ────────────────────────
  useEffect(() => {
    Object.entries(sensorActive).forEach(([id, active]) => {
      window.PumpSmartAPI.setSensorConnected(id, active);
    });
    window.PumpSmartAPI.setCluster(CLUSTER_IDS[activeCluster]);
  }, [sensorActive, activeCluster]);

  // ── History tab — buffer state poll (settings card) ──
  useEffect(() => {
    if (!onboardingDone) return;
    const fetchState = async () => {
      try {
        const st = await window.PumpSmartAPI.getSensorHistoryState();
        setHistoryBufferState(st);
      } catch (e) {}
    };
    fetchState();
    const t = setInterval(fetchState, 10000);
    return () => clearInterval(t);
  }, [onboardingDone]);

  const handleAcknowledge = async () => {
    try {
      await window.PumpSmartAPI.acknowledge("Operator acknowledged — investigation in progress.");
      setShowPopup(false);
    } catch(e) { setShowPopup(false); }
  };

  const handleLoadHistoryTab = async () => {
    setHistoryTabLoading(true);
    try {
      const data = await window.PumpSmartAPI.loadHistoryTab(historyTabRange);
      setHistoryTabData(data);
    } catch (e) {
      console.error("history load failed", e);
    }
    setHistoryTabLoading(false);
  };

  const handleTabClick = (tabId) => {
    if (!onboardingDone && GATED_TABS.includes(tabId)) { setTab("guide"); return; }
    if (tabId === "predictions") setVerdictSent(null);
    setTab(tabId);
  };

  const toggleChartChannel = (chName) => {
    setChartChannels(prev =>
      prev.includes(chName) ? prev.filter(c => c !== chName) : [...prev, chName]);
  };

  const accentColor = ALERT_COLORS[alertState] || "#00e676";

  const sideItems = [
    { id:"dashboard",   icon:"🏠", label:"Dashboard" },
    { id:"sensor",      icon:"🔌", label:"Sensor Plugin" },
    { id:"analytics",   icon:"📈", label:"Analytics" },
    { id:"predictions", icon:"🧠", label:"Predictions" },
    { id:"history",     icon:"🕒", label:"History" },
    { id:"settings",    icon:"⚙️",  label:"Settings" },
    { id:"guide",       icon:"📖", label:"Guide & Disclaimer", accent:true },
  ];

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ display:"flex", background:"#04101e", minHeight:"100vh",
                  fontFamily:"'Inter',system-ui,sans-serif", color:"#c0d8f0",
                  fontSize:13, position:"relative", overflow:"hidden" }}>
      <div style={{ position:"absolute", inset:0, pointerEvents:"none", zIndex:0,
        background:"radial-gradient(ellipse 90% 55% at 50% -5%, rgba(0,90,200,.2) 0%,transparent 65%)" }}/>

      {showOnboarding && (
        <OnboardingPopup
          onAccept={() => { setOnboardingDone(true); setShowOnboarding(false); setTab("dashboard"); }}
          onViewGuide={() => { setShowOnboarding(false); setTab("guide"); }}
        />
      )}

      {/* Sidebar */}
      <div style={{ width:196, flexShrink:0, background:"rgba(4,12,24,.95)",
                    borderRight:"1px solid rgba(20,55,95,.5)", display:"flex",
                    flexDirection:"column", position:"relative", zIndex:2 }}>
        <div style={{ padding:"18px 16px 14px", borderBottom:"1px solid rgba(20,55,95,.5)" }}>
          <div style={{ fontSize:20, fontWeight:700, color:"#fff", letterSpacing:1 }}>
            ⚡ Pump<span style={{ color:"#00d4ff" }}>Smart</span>
          </div>
          <div style={{ fontSize:10, color:"#1a5060", marginTop:2, letterSpacing:1 }}>INDUSTRIAL MONITOR v14.2</div>
        </div>

        <div style={{ margin:"10px 10px", background:"rgba(0,28,56,.55)",
                      border:"1px solid rgba(20,65,105,.45)", borderRadius:8, padding:"10px 12px" }}>
          <div style={{ fontSize:11, color:"#00d4ff", fontWeight:600 }}>🏭 PUMP-0032</div>
          <div style={{ fontSize:9, color:"#1a5060", fontStyle:"italic", marginTop:1 }}>(single-pump v14.2)</div>
          <div style={{ fontSize:10, color:"#1a5060", marginTop:2 }}>110 kW · 7-stage · 40 bar</div>
          <div style={{ display:"flex", alignItems:"center", gap:6, marginTop:7 }}>
            <div style={{ width:7, height:7, borderRadius:"50%",
              background:connectedCount>0?"#00e676":"#ff8800",
              boxShadow:`0 0 7px ${connectedCount>0?"#00e676":"#ff8800"}` }}/>
            <span style={{ fontSize:10, color:connectedCount>0?"#00e676":"#ff8800" }}>
              {connectedCount>0 ? `${connectedCount}/8 channels active` : "No channels active"}
            </span>
          </div>
          {lastUpdate && <div style={{ fontSize:9, color:"#1a4050", marginTop:4 }}>Updated: {lastUpdate}</div>}
        </div>

        <nav style={{ flex:1, padding:"6px 0" }}>
          {sideItems.map(item => {
            const isGated  = !onboardingDone && GATED_TABS.includes(item.id);
            const isActive = tab === item.id;
            return (
              <div key={item.id} onClick={() => handleTabClick(item.id)}
                style={{ display:"flex", alignItems:"center", gap:10, padding:"11px 16px",
                  cursor:"pointer", transition:"all .2s", opacity:isGated?0.45:1,
                  borderLeft:`3px solid ${isActive?"#00d4ff":item.accent?"rgba(0,180,220,.15)":"transparent"}`,
                  background:isActive?"rgba(0,180,220,.07)":item.accent?"rgba(0,160,200,.03)":"transparent",
                  marginTop:item.accent?4:0, borderTop:item.accent?"1px solid rgba(20,55,95,.35)":"none" }}>
                <span style={{ fontSize:15, color:isActive?"#00d4ff":"#2a5a70" }}>{item.icon}</span>
                <span style={{ fontSize:12, color:isActive?"#00d4ff":"#3a6a7a",
                  fontWeight:isActive?500:400, flex:1 }}>{item.label}</span>
                {isGated && <span style={{ fontSize:10 }}>🔒</span>}
                {item.id==="sensor" && !isGated && noSensors && (
                  <span style={{ width:6, height:6, borderRadius:"50%",
                    background:"#ff8800", boxShadow:"0 0 5px #ff8800" }}/>
                )}
              </div>
            );
          })}
        </nav>

        <div style={{ padding:"12px 14px", borderTop:"1px solid rgba(20,55,95,.4)" }}>
          <div style={{ fontSize:10, color:"#1a4050", lineHeight:1.7 }}>
            ⚠️ Advisory only<br/>Verify predictions physically<br/>Not for autonomous control
          </div>
        </div>
      </div>

      {/* Main */}
      <div style={{ flex:1, display:"flex", flexDirection:"column", zIndex:1, minWidth:0, position:"relative" }}>
        {/* Topbar */}
        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between",
                      padding:"10px 18px", background:"rgba(4,12,24,.9)",
                      borderBottom:"1px solid rgba(20,55,95,.5)" }}>
          <div style={{ display:"flex", alignItems:"center", gap:12 }}>
            <span style={{ fontSize:15, fontWeight:600, color:"#e0f0ff" }}>
              {({ dashboard:"🏠 Dashboard", sensor:"🔌 Sensor Plugin",
                  analytics:"📈 Analytics", predictions:"🧠 Predictions",
                  history:"🕒 History", settings:"⚙️ Settings",
                  guide:"📖 Guide & Disclaimer" })[tab] || "🏠 Dashboard"}
            </span>
            {onboardingDone && connectedCount > 0 && (
              <div style={{ display:"flex", alignItems:"center", gap:6, padding:"4px 12px",
                borderRadius:20, background:`${accentColor}14`, border:`1px solid ${accentColor}40` }}>
                <div style={{ width:6, height:6, borderRadius:"50%",
                  background:accentColor, boxShadow:`0 0 7px ${accentColor}` }}/>
                <span style={{ fontSize:11, fontWeight:600, color:accentColor }}>{alertState}</span>
              </div>
            )}
            {onboardingDone && noSensors && (
              <div style={{ fontSize:11, color:"#ff8800", background:"rgba(255,136,0,.08)",
                padding:"3px 10px", borderRadius:6, border:"1px solid rgba(255,136,0,.2)" }}>
                🔌 No data source active — inference paused
              </div>
            )}
            {apiError && (
              <div style={{ fontSize:11, color:"#ff4466", background:"rgba(255,50,50,.08)",
                padding:"3px 10px", borderRadius:6, border:"1px solid rgba(255,50,50,.2)" }}>
                ⚠️ {apiError}
              </div>
            )}
          </div>
          {onboardingDone && (
            <button onClick={() => { setOnboardingDone(false); setTab("guide"); }}
              style={{ padding:"5px 12px", background:"rgba(20,55,95,.3)",
                border:"1px solid rgba(40,90,130,.4)", color:"#3a6a7a",
                borderRadius:8, cursor:"pointer", fontSize:11 }}>
              🔒 Re-acknowledge
            </button>
          )}
        </div>

        {/* Content */}
        <div style={{ flex:1, padding:14, overflow:"auto", position:"relative",
                      display:"flex", flexDirection:"column" }}>

          {/* ── DASHBOARD ── */}
          {tab==="dashboard" && onboardingDone && (
            noSensors ? <SensorGate onGoToSensor={() => setTab("sensor")}/> : (
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 255px", gap:12 }}>
                {/* Vitals */}
                <Card style={{ gridColumn:"1/3" }} animate>
                  <CardHdr title="💓 Pump Vitals" right={
                    <span style={{ fontSize:11, color:prediction?"#00e676":"#ff8800" }}>
                      {prediction?"🟢 Live data":"⏳ Awaiting first prediction..."}
                    </span>}/>
                  {prediction ? (
                    <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:10 }}>
                      {[
                        { label:"Alert State",    val:alertState,                          ok:alertState==="NORMAL" },
                        { label:"Score A (L1)",   val:prediction.score_A?.toFixed(4),     ok:(prediction.score_A||0)<0.15 },
                        { label:"CUSUM S_n (L3)", val:prediction.cusum_Sn?.toFixed(3),   ok:(prediction.cusum_Sn||0)<2.0 },
                        { label:"Confidence",     val:(prediction.confidence_pct?.toFixed(1)||"—")+"%", ok:(prediction.confidence_pct||0)>=70 },
                      ].map((v,i) => (
                        <div key={i} style={{ background:"rgba(4,12,26,.75)",
                          border:"1px solid rgba(20,60,100,.4)", borderRadius:10, padding:"12px 14px" }}>
                          <div style={{ fontSize:10, color:"#2a5a70", marginBottom:5 }}>{v.label}</div>
                          <div style={{ fontSize:24, fontWeight:700,
                            color:v.ok?"#e0f0ff":accentColor, lineHeight:1 }}>{v.val??"—"}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ padding:"24px 0", textAlign:"center", color:"#2a5a70" }}>
                      Data will populate once the inference pipeline processes the first window.
                    </div>
                  )}
                </Card>

                {/* Risk Hub */}
                <Card style={{ gridColumn:3, gridRow:"1/3", display:"flex", flexDirection:"column", alignItems:"center" }} animate>
                  <CardHdr title="🎯 AI Risk Hub"/>
                  <RiskGauge alertState={alertState}/>
                  {prediction && (
                    <div style={{ width:"100%", display:"grid", gridTemplateColumns:"1fr 1fr", gap:7, marginTop:12 }}>
                      {[
                        { k:"Score A",  v:prediction.score_A?.toFixed(4),   c:accentColor },
                        { k:"Score B",  v:prediction.score_B?.toFixed(4),   c:"#3a7a9a" },
                        { k:"Score C",  v:prediction.score_C?.toFixed(4),   c:"#3a7a9a" },
                        { k:"Fault",    v:prediction.fault_label,            c:"#c0d8f0", small:true },
                      ].map((it,i) => (
                        <div key={i} style={{ background:"rgba(4,12,26,.75)",
                          border:"1px solid rgba(20,60,100,.35)", borderRadius:8, padding:"8px 10px" }}>
                          <div style={{ fontSize:10, color:"#2a5060", marginBottom:3 }}>{it.k}</div>
                          <div style={{ fontSize:it.small?11:15, fontWeight:600, color:it.c, lineHeight:1.2 }}>
                            {it.v??"—"}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>

                {/* ── LIVE CHART — server-side history, last 5 min, 1s polling ── */}
                <Card style={{ gridColumn:"1/3", gridRow:2 }} animate>
                  <CardHdr title="📡 Live Sensor Trend — last 5 minutes (server-side history)"
                    right={
                      <div style={{ display:"flex", gap:6, flexWrap:"wrap", maxWidth:380, justifyContent:"flex-end" }}>
                        {SENSOR_CHANNELS.map(s => (
                          <button key={s.id} onClick={() => toggleChartChannel(s.ch)}
                            style={{ fontSize:9, padding:"2px 7px", borderRadius:5, cursor:"pointer",
                              background:chartChannels.includes(s.ch) ? `${s.color}22` : "transparent",
                              border:`1px solid ${chartChannels.includes(s.ch) ? s.color+"66" : "rgba(20,55,95,.3)"}`,
                              color:chartChannels.includes(s.ch) ? s.color : "#2a5060" }}>
                            {s.ch}
                          </button>
                        ))}
                      </div>}/>
                  <SensorHistoryChart historyData={liveChartData}
                                       channelsToShow={chartChannels} height={200}/>
                </Card>

                {/* Fault classification */}
                <Card style={{ gridColumn:"1/2", gridRow:3 }} animate>
                  <CardHdr title="🔍 Fault Classification"/>
                  {prediction ? (
                    <div>
                      <div style={{ fontSize:10, color:"#3a6a7a", marginBottom:4 }}>Primary fault</div>
                      <div style={{ fontSize:22, fontWeight:700, color:accentColor, marginBottom:10 }}>
                        {prediction.fault_label || "—"}
                      </div>
                      <div style={{ fontSize:10, color:"#3a6a7a", marginBottom:4 }}>
                        Confidence — {prediction.confidence_pct?.toFixed(1)}%
                      </div>
                      <div style={{ height:6, background:"rgba(20,60,100,.4)", borderRadius:3, marginBottom:10 }}>
                        <div style={{ width:`${prediction.confidence_pct||0}%`, height:6,
                          background:accentColor, borderRadius:3 }}/>
                      </div>
                      {prediction.unknown_fault_flag && (
                        <div style={{ padding:"8px 12px", background:"rgba(255,136,0,.08)",
                          border:"1px solid rgba(255,136,0,.25)", borderRadius:8,
                          fontSize:11, color:"#ff8800" }}>
                          ⚠️ UNKNOWN FAULT — confidence below 70%. Inspect regardless.
                        </div>
                      )}
                      {prediction.causal_chain && (
                        <div style={{ marginTop:8, padding:"8px 12px",
                          background:"rgba(0,180,220,.07)", border:"1px solid rgba(0,180,220,.2)",
                          borderRadius:8, fontSize:11, color:"#6ab0cc" }}>
                          🔗 {prediction.causal_chain}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div style={{ fontSize:12, color:"#2a5060", padding:"12px 0" }}>
                      Awaiting first prediction…
                    </div>
                  )}
                </Card>

                {/* CUSUM */}
                <Card style={{ gridColumn:"2/3", gridRow:3 }} animate>
                  <CardHdr title="📈 CUSUM Accumulator S_n"
                    right={<span style={{ fontSize:10, color:"#2a5060", padding:"2px 7px",
                      background:"rgba(4,12,26,.5)", border:"1px solid rgba(20,60,100,.3)",
                      borderRadius:6 }}>L3 · Gradual wear</span>}/>
                  {healthData ? (
                    <div>
                      <div style={{ display:"flex", gap:8, marginBottom:10 }}>
                        {[
                          { k:"S_n",   v:healthData.cusum_state?.cusum_Sn?.toFixed(4)||"0.0000",
                            c:(healthData.cusum_state?.cusum_Sn||0)>4?"#ff2244":
                               (healthData.cusum_state?.cusum_Sn||0)>2?"#ffcc00":"#00e676" },
                          { k:"H (alarm)", v:"5.0", c:"#3a6a7a" },
                          { k:"State",  v:healthData.cusum_state?.cusum_alert||"NORMAL", c:accentColor },
                        ].map((it,i) => (
                          <div key={i} style={{ flex:1, background:"rgba(4,12,26,.75)",
                            border:"1px solid rgba(20,60,100,.35)", borderRadius:8, padding:"7px 10px" }}>
                            <div style={{ fontSize:10, color:"#2a5060", marginBottom:3 }}>{it.k}</div>
                            <div style={{ fontSize:16, fontWeight:700, color:it.c }}>{it.v}</div>
                          </div>
                        ))}
                      </div>
                      <div style={{ fontSize:11, color:"#2a5060" }}>
                        θ_t: {healthData.rolling_state?.theta_t?.toFixed(6)}
                        {healthData.rolling_state?.drift_locked &&
                          <span style={{ color:"#ff8800" }}> · DRIFT LOCKED</span>}
                      </div>
                    </div>
                  ) : (
                    <div style={{ fontSize:12, color:"#2a5060" }}>Connecting to server…</div>
                  )}
                </Card>

                {/* Event log */}
                <Card style={{ gridColumn:3, gridRow:"3/5" }} animate>
                  <CardHdr title="🚨 Event Log"
                    right={<span style={{ fontSize:11, color:"#00b4d8", cursor:"pointer" }}
                      onClick={() => setTab("history")}>All →</span>}/>
                  {eventLog.length===0 ? (
                    <div style={{ fontSize:12, color:"#2a5060", padding:"12px 0" }}>
                      Events appear here as predictions arrive.
                    </div>
                  ) : (
                    <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                      {eventLog.slice(0,6).map((ev,i) => {
                        const ec = ALERT_COLORS[ev.state]||"#3a6a7a";
                        return (
                          <div key={i} style={{ display:"flex", gap:8, padding:"8px 10px",
                            borderRadius:8, background:"rgba(4,12,26,.65)",
                            border:"1px solid rgba(20,60,100,.35)", alignItems:"flex-start" }}>
                            <div style={{ width:26, height:26, borderRadius:6, flexShrink:0,
                              background:`${ec}18`, border:`1px solid ${ec}44`,
                              display:"flex", alignItems:"center", justifyContent:"center",
                              fontSize:12, color:ec }}>
                              {ev.state==="DANGER"?"🔴":ev.state==="WARN"?"⚠️":ev.state==="WATCH"?"👁":"✅"}
                            </div>
                            <div style={{ flex:1 }}>
                              <div style={{ display:"flex", gap:5, marginBottom:2 }}>
                                <span style={{ fontSize:10, color:"#00b4d8" }}>{ev.time}</span>
                                <span style={{ fontSize:10, padding:"1px 6px", borderRadius:6,
                                  background:`${ec}18`, color:ec }}>{ev.state}</span>
                              </div>
                              <div style={{ fontSize:11, color:"#7aabb8" }}>{ev.fault} · {ev.conf}</div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </Card>
              </div>
            )
          )}

          {/* ── SENSOR PLUGIN ── */}
          {tab==="sensor" && onboardingDone && (
            <div>
              <div style={{ marginBottom:16, padding:"16px 20px",
                background:"rgba(0,100,180,.08)", border:"1px solid rgba(0,140,220,.2)",
                borderRadius:12, lineHeight:1.8 }}>
                <div style={{ fontSize:13, fontWeight:600, color:"#00d4ff", marginBottom:10 }}>
                  🔌 Sensor data connection guide
                </div>
                <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
                  {[
                    { title:"M12 adversarial test (local)",  c:"#00e676",
                      body:`Run: python src/module_12b_adversarial_runner.py --mode smoke\n\nThis generates physics-synthetic fault sequences and POSTs normalised 50×8 windows to POST /api/anomaly_detect. Mark the relevant channels as "Data active" below.` },
                    { title:"Real SCADA/PLC/DAQ connection", c:"#00b4d8",
                      body:`POST to: http://your-server:8000/api/anomaly_detect\nContent-Type: application/json\n\nBody: {\n  "window": [[...50 rows × 8 columns of M3-normalised values...]],\n  "pump_id": "PUMP-0032",\n  "cluster": "steady_state"\n}\n\nChannel column order (index 0→7):\n  0: Mot.SV  1: Pmp.SV  2: Mot.TV  3: Pmp.PV\n  4: Temp.SV  5: Pres.SV  6: Pmp.TV  7: Mot.PV\n\nAll values must be M3 cluster-normalised before sending.\nNormal operation → values in 0–1. Faults → drift above 1.0.` },
                    { title:"Python HTTP client example",    c:"#9b7fe8",
                      body:`import requests, numpy as np\n\nwindow = np.random.normal(0.5, 0.05, (50, 8)).tolist()  # replace with real data\n\nresponse = requests.post(\n    "http://127.0.0.1:8000/api/anomaly_detect",\n    json={"window": window, "pump_id": "PUMP-0032", "cluster": "steady_state"}\n)\nprint(response.json()["fault_label"], response.json()["confidence_pct"])` },
                  ].map((item,i) => (
                    <div key={i} style={{ padding:"12px 16px",
                      background:"rgba(4,12,26,.65)", borderLeft:`3px solid ${item.c}`,
                      border:`1px solid ${item.c}22`, borderRadius:10 }}>
                      <div style={{ fontSize:12, fontWeight:600, color:item.c, marginBottom:6 }}>{item.title}</div>
                      <pre style={{ fontSize:11, color:"#5a8aaa", lineHeight:1.7,
                        whiteSpace:"pre-wrap", fontFamily:"monospace", margin:0 }}>{item.body}</pre>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ marginBottom:14, background:"rgba(6,18,36,.88)",
                border:"1px solid rgba(20,60,100,.5)", borderRadius:12, padding:"12px 16px" }}>
                <div style={{ fontSize:11, color:"#5a9ab0", fontWeight:500,
                  letterSpacing:.8, textTransform:"uppercase", marginBottom:10 }}>
                  Active operating cluster (used in inference window generation)
                </div>
                <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
                  {CLUSTER_LABELS.map((label,i) => (
                    <button key={i} onClick={() => setActiveCluster(i)}
                      style={{ padding:"6px 14px", borderRadius:8, cursor:"pointer",
                        fontSize:12, fontWeight:activeCluster===i?600:400,
                        border:`1px solid ${activeCluster===i?CLUSTER_COLORS[i]+"88":"rgba(20,55,95,.5)"}`,
                        background:activeCluster===i?CLUSTER_COLORS[i]+"18":"transparent",
                        color:activeCluster===i?CLUSTER_COLORS[i]:"#3a6a7a" }}>
                      {label}{activeCluster===i?" ◀":""}
                    </button>
                  ))}
                </div>
                {activeCluster===0 && <div style={{ marginTop:8, fontSize:11, color:"#ffbb00" }}>⚠️ Startup: vibration 2–4× higher — shaft resonance, NOT a fault.</div>}
                {activeCluster===3 && <div style={{ marginTop:8, fontSize:11, color:"#00b4d8" }}>ℹ️ Cooldown: casing temp may drop below ambient ≤1.4°C — flash evaporation (C-09), not a fault.</div>}
              </div>

              <div style={{ marginBottom:10, fontSize:12, color:"#3a6a7a" }}>
                Mark channels as "Data active" once your data source is running and sending data
                for that channel. This tells the inference pipeline which channels have live data.
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
                {SENSOR_CHANNELS.map((s) => {
                  const active = sensorActive[s.id];
                  return (
                    <div key={s.id} style={{ background:"rgba(6,18,36,.88)", borderRadius:12, padding:16,
                      border:`1px solid ${active?"rgba(0,200,120,.2)":"rgba(40,70,100,.35)"}` }}>
                      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
                        <div>
                          <div style={{ fontSize:13, fontWeight:500, color:"#e0f0ff", marginBottom:2 }}>
                            {s.icon} {s.name}
                          </div>
                          <div style={{ fontSize:10, color:"#2a5a70", marginBottom:2 }}>{s.desc}</div>
                          <div style={{ fontSize:10, color:"#2a5060" }}>
                            API channel: <code style={{ color:"#00d4ff" }}>{s.ch}</code>{" "}
                            (column index {s.idx}) · Unit: {s.unit}
                          </div>
                        </div>
                        <div style={{ display:"flex", alignItems:"flex-start", gap:6, marginLeft:10, flexShrink:0 }}>
                          <div style={{ width:8, height:8, borderRadius:"50%", marginTop:2,
                            background:active?"#00e676":"#555",
                            boxShadow:active?"0 0 7px #00e676":"none" }}/>
                          <span style={{ fontSize:10, color:active?"#00e676":"#3a5060" }}>
                            {active?"Data active":"Inactive"}
                          </span>
                        </div>
                      </div>
                      <button onClick={() => setSensorActive(p => ({ ...p, [s.id]:!p[s.id] }))}
                        style={{ width:"100%", padding:"8px 0", borderRadius:7, cursor:"pointer",
                          fontSize:11, fontWeight:600,
                          border:`1px solid ${active?"rgba(255,100,0,.4)":"rgba(0,200,120,.35)"}`,
                          background:active?"rgba(255,100,0,.08)":"rgba(0,200,120,.1)",
                          color:active?"#ff8800":"#00e676" }}>
                        {active?"Mark as inactive (data stopped)":"Mark as active (data flowing)"}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── ANALYTICS ── */}
          {tab==="analytics" && onboardingDone && (
            noSensors ? <SensorGate onGoToSensor={() => setTab("sensor")}/> : (
              <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
                {/* Channel trend chart — last 1 hour with score_A + CUSUM overlay */}
                <Card animate>
                  <CardHdr title="📈 Sensor Trends — last 1 hour (server-side, polled every 5s)"
                    right={
                      <div style={{ display:"flex", gap:6, flexWrap:"wrap", maxWidth:380, justifyContent:"flex-end" }}>
                        {SENSOR_CHANNELS.map(s => (
                          <button key={s.id} onClick={() => toggleChartChannel(s.ch)}
                            style={{ fontSize:9, padding:"2px 7px", borderRadius:5, cursor:"pointer",
                              background:chartChannels.includes(s.ch) ? `${s.color}22` : "transparent",
                              border:`1px solid ${chartChannels.includes(s.ch) ? s.color+"66" : "rgba(20,55,95,.3)"}`,
                              color:chartChannels.includes(s.ch) ? s.color : "#2a5060" }}>
                            {s.ch}
                          </button>
                        ))}
                      </div>}/>
                  <SensorHistoryChart historyData={analyticsData}
                                       channelsToShow={chartChannels}
                                       showDerived={true} height={300}/>
                </Card>

                <Card animate>
                  <CardHdr title="📊 Model Layer Outputs — current prediction"/>
                  {prediction ? (
                    <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:12 }}>
                      {[
                        { label:"L1 score_A",           val:prediction.score_A,             note:"→ L4 adaptive threshold (q=0.110058 locked)" },
                        { label:"L2 score_B (drift)",   val:prediction.score_B,             note:"→ L3 CUSUM only — Invariant 19" },
                        { label:"L2 score_C (chain)",   val:prediction.score_C,             note:"→ M7 XGBoost only — Invariant 19" },
                        { label:"L3 CUSUM S_n",         val:prediction.cusum_Sn,            note:"Gradual wear accumulator — H=5.0" },
                        { label:"L4 θ_t",               val:prediction.adaptive_threshold,  note:"Rolling 6-hr baseline — lock at 1.5×θ_initial" },
                        { label:"OOD Mahalanobis",      val:prediction.mahal_dist,          note:`tau_p99: ${prediction.mahal_dist !== undefined ? "15.0" : "—"}` },
                      ].map((item,i) => (
                        <div key={i} style={{ background:"rgba(4,12,26,.75)",
                          border:"1px solid rgba(20,60,100,.4)", borderRadius:10, padding:"12px 14px" }}>
                          <div style={{ fontSize:10, color:"#2a5060", marginBottom:5 }}>{item.label}</div>
                          <div style={{ fontSize:24, fontWeight:700, color:"#00d4ff", marginBottom:4 }}>
                            {item.val?.toFixed(4) ?? "—"}
                          </div>
                          <div style={{ fontSize:10, color:"#3a6a7a" }}>{item.note}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <EmptyState icon="📊" message="Awaiting first prediction"
                      sub="Start data source and mark channels as active in Sensor Plugin"/>
                  )}
                </Card>

                {prediction?.top_shap_features && Object.keys(prediction.top_shap_features).length > 0 && (
                  <Card animate>
                    <CardHdr title="🔬 Feature Importance (XGBoost gain — top 3)"/>
                    <div style={{ fontSize:11, color:"#2a5060", marginBottom:12, lineHeight:1.6 }}>
                      Relative gain contribution to current prediction. Higher = more influence on classification.
                      Feature names refer to M6.5r 33-feature schema columns.
                    </div>
                    {(() => {
                      const entries = Object.entries(prediction.top_shap_features);
                      const maxVal  = Math.max(...entries.map(([,v]) => v));
                      return entries.map(([k,v],i) => (
                        <div key={i} style={{ display:"flex", alignItems:"center", gap:12, marginBottom:10 }}>
                          <div style={{ width:80, fontSize:12, color:"#8ab0c8", fontFamily:"monospace" }}>{k}</div>
                          <div style={{ flex:1, height:6, background:"rgba(20,55,95,.4)", borderRadius:3 }}>
                            <div style={{ width:`${(v/maxVal)*100}%`, height:6,
                              background:"#ff8800", borderRadius:3 }}/>
                          </div>
                          <div style={{ width:60, fontSize:11, color:"#ff8800",
                            textAlign:"right" }}>{((v/maxVal)*100).toFixed(1)}%</div>
                        </div>
                      ));
                    })()}
                  </Card>
                )}
              </div>
            )
          )}

          {/* ── PREDICTIONS ── */}
          {tab==="predictions" && onboardingDone && (
            noSensors ? <SensorGate onGoToSensor={() => setTab("sensor")}/> : (
              <Card animate>
                <CardHdr title="🧠 7-Field Output — latest prediction (real-time)"/>
                {prediction ? (
                  <>
                    <div style={{ display:"flex", flexDirection:"column", gap:0 }}>
                      {[
                        { n:"01", k:"🏷️ Primary Classification",   v:prediction.fault_label, c:"#00d4ff" },
                        { n:"02", k:"📊 Confidence",                 v:`${prediction.confidence_pct?.toFixed(1)}%${prediction.unknown_fault_flag?" ⚠️ LOW":""}`,
                          c:(prediction.confidence_pct||0)>85?"#00e676":(prediction.confidence_pct||0)>70?"#ff8800":"#ff2244" },
                        { n:"03", k:"🔬 Physical Interpretation",   v:prediction.probable_physical_condition },
                        { n:"04", k:"📡 Expected Sensor Behaviour", v:prediction.expected_sensor_behavior },
                        { n:"05", k:"⏳ Consequence if Ignored",    v:prediction.operational_risk_if_ignored,
                          c:["WARN","DANGER"].includes(alertState)?"#ff8800":"#00e676" },
                        { n:"06", k:"🔧 Recommended Action",        v:prediction.recommended_action },
                        { n:"07", k:"⚠️ Model Disclaimer",          v:prediction.model_limitation_disclaimer, c:"#3a6a7a" },
                      ].map((row,i) => (
                        <div key={i} style={{ display:"grid", gridTemplateColumns:"40px 200px 1fr",
                          gap:16, padding:"12px 0", borderBottom:"1px solid rgba(20,55,95,.3)" }}>
                          <div style={{ fontSize:12, color:"#2a5060", fontFamily:"monospace" }}>{row.n}</div>
                          <div style={{ fontSize:11, color:"#3a6a7a", fontWeight:500, lineHeight:1.4 }}>{row.k}</div>
                          <div style={{ fontSize:13, color:row.c||"#c0d8f0", lineHeight:1.6 }}>{row.v||"—"}</div>
                        </div>
                      ))}
                    </div>

                    {prediction.m8p6_addendum?.triggered && (
                      <div style={{ marginTop:10, padding:"10px 14px",
                        background:"rgba(230,126,34,.07)", border:"1px solid rgba(230,126,34,.3)",
                        borderLeft:"3px solid #e67e22", borderRadius:9 }}>
                        <div style={{ fontSize:11, fontWeight:600, color:"#e67e22", marginBottom:4 }}>
                          ⚠️ M8p6 Sensor Health Addendum — Field 6 annotation (C-28)
                        </div>
                        <div style={{ fontSize:12, color:"#d4874a", lineHeight:1.6 }}>
                          {prediction.m8p6_addendum.addendum_text}
                        </div>
                        <div style={{ fontSize:10, color:"#7a5030", marginTop:4 }}>
                          Annotation only. Does not override fault label or confidence. override_existing_prediction: false (locked).
                        </div>
                      </div>
                    )}

                    <div style={{ marginTop:14, padding:"12px 14px",
                      background:"rgba(4,12,26,.5)", border:"1px solid rgba(20,55,95,.4)", borderRadius:10 }}>
                      <div style={{ fontSize:11, color:"#3a6a7a", marginBottom:8, lineHeight:1.5 }}>
                        📋 <strong style={{ color:"#8ab0c8" }}>Rate this prediction</strong> — after
                        physically verifying, submit your verdict. This is the ONLY point where
                        active-learning data is written.
                      </div>
                      {verdictSent ? (
                        <div style={{ padding:"10px 14px", background:"rgba(0,200,100,.07)",
                          border:"1px solid rgba(0,200,100,.25)", borderRadius:9,
                          fontSize:13, color:"#00e676" }}>
                          ✅ Verdict recorded: <strong>{verdictSent}</strong>
                        </div>
                      ) : (
                        <div style={{ display:"flex", gap:10 }}>
                          {[
                            { l:"✅ Correct",   c:"#00e676", bg:"rgba(0,200,100,.1)",  val:"CORRECT" },
                            { l:"❌ Incorrect", c:"#ff4466", bg:"rgba(200,30,60,.08)", val:"INCORRECT" },
                            { l:"❓ Unsure",    c:"#ffcc00", bg:"rgba(200,180,0,.08)", val:"UNSURE" },
                          ].map((btn,i) => (
                            <button key={i} onClick={async () => {
                                try { await window.PumpSmartAPI.submitVerdict(prediction.prediction_id, btn.val); }
                                catch(e) {}
                                setVerdictSent(btn.val);
                              }}
                              style={{ flex:1, padding:"10px 6px", borderRadius:8, cursor:"pointer",
                                fontSize:12, fontWeight:500, border:`1px solid ${btn.c}44`,
                                background:btn.bg, color:btn.c }}>{btn.l}</button>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <EmptyState icon="🧠" message="No prediction yet"
                    sub="The 7-field output will appear here once data is flowing and the model has processed a window."/>
                )}
              </Card>
            )
          )}

          {/* ── HISTORY ── on-demand sensor history chart + event log */}
          {tab==="history" && onboardingDone && (
            noSensors ? <SensorGate onGoToSensor={() => setTab("sensor")}/> : (
              <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
                <Card animate>
                  <CardHdr title="📊 Sensor History (server-side ring buffer — multi-client safe)"
                    right={
                      <div style={{ display:"flex", gap:6, alignItems:"center" }}>
                        {[1, 6, 24].map(hr => (
                          <button key={hr} onClick={() => setHistoryTabRange(hr)}
                            style={{ fontSize:10, padding:"4px 10px", borderRadius:6,
                              cursor:"pointer",
                              background:historyTabRange===hr?"rgba(0,180,220,.2)":"transparent",
                              border:`1px solid ${historyTabRange===hr?"rgba(0,180,220,.4)":"rgba(20,55,95,.4)"}`,
                              color:historyTabRange===hr?"#00d4ff":"#3a6a7a" }}>
                            {hr}h
                          </button>
                        ))}
                        <button onClick={handleLoadHistoryTab} disabled={historyTabLoading}
                          style={{ fontSize:11, padding:"5px 12px", borderRadius:7, cursor:"pointer",
                            background:"rgba(0,180,220,.15)",
                            border:"1px solid rgba(0,180,220,.4)", color:"#00d4ff", fontWeight:600 }}>
                          {historyTabLoading ? "⏳ Loading…" : "📥 Load"}
                        </button>
                      </div>}/>

                  <div style={{ marginBottom:10, padding:"8px 12px",
                    background:"rgba(4,12,26,.5)", border:"1px solid rgba(20,55,95,.3)",
                    borderRadius:8, fontSize:11, color:"#5a8aaa", lineHeight:1.6 }}>
                    ℹ️ History persists across alarm acknowledgements (forensic retention).
                    Server-side ring buffer holds up to 24 hours at 1 Hz.
                    {historyBufferState && (
                      <span style={{ marginLeft:8, color:"#6ab0cc" }}>
                        · Buffer: {historyBufferState.buffer_fill}/{historyBufferState.buffer_capacity}
                        {" "}({historyBufferState.fill_pct}%)
                      </span>
                    )}
                  </div>

                  <div style={{ display:"flex", gap:6, flexWrap:"wrap", marginBottom:10 }}>
                    {SENSOR_CHANNELS.map(s => (
                      <button key={s.id} onClick={() => toggleChartChannel(s.ch)}
                        style={{ fontSize:10, padding:"3px 9px", borderRadius:5, cursor:"pointer",
                          background:chartChannels.includes(s.ch) ? `${s.color}22` : "transparent",
                          border:`1px solid ${chartChannels.includes(s.ch) ? s.color+"66" : "rgba(20,55,95,.3)"}`,
                          color:chartChannels.includes(s.ch) ? s.color : "#2a5060" }}>
                        {s.ch}
                      </button>
                    ))}
                  </div>

                  <SensorHistoryChart historyData={historyTabData}
                                       channelsToShow={chartChannels}
                                       showDerived={true} height={320}/>
                </Card>

                <Card animate>
                  <CardHdr title="🕒 Event History — this session"
                    right={<span style={{ fontSize:10, color:"#2a5060" }}>
                      {eventLog.length} event(s) · resets on page reload
                    </span>}/>
                  {eventLog.length===0 ? (
                    <EmptyState icon="📋" message="No events recorded in this session"
                      sub="Events accumulate here as predictions arrive."/>
                  ) : (
                    <>
                      <div style={{ display:"grid",
                        gridTemplateColumns:"90px 90px 90px 1fr 70px 80px",
                        gap:10, padding:"0 0 8px", borderBottom:"1px solid rgba(20,55,95,.4)",
                        fontSize:10, color:"#2a5060", textTransform:"uppercase", letterSpacing:.7 }}>
                        <span>Date</span><span>Time</span><span>State</span>
                        <span>Fault</span><span>Conf</span><span>Score A</span>
                      </div>
                      {eventLog.map((ev,i) => {
                        const sc = ALERT_COLORS[ev.state]||"#3a6a7a";
                        return (
                          <div key={i} style={{ display:"grid",
                            gridTemplateColumns:"90px 90px 90px 1fr 70px 80px",
                            gap:10, padding:"9px 0", borderBottom:"1px solid rgba(20,55,95,.2)",
                            fontSize:12, alignItems:"center" }}>
                            <div style={{ color:"#5a9ab0", fontFamily:"monospace", fontSize:10 }}>{ev.date}</div>
                            <div style={{ color:"#00b4d8", fontFamily:"monospace", fontSize:10 }}>{ev.time}</div>
                            <div><span style={{ fontSize:10, padding:"2px 7px", borderRadius:6,
                              background:`${sc}18`, color:sc, border:`1px solid ${sc}30`,
                              fontWeight:600 }}>{ev.state}</span></div>
                            <div style={{ color:"#c0d8f0" }}>{ev.fault}</div>
                            <div style={{ color:"#8ab0c8", fontFamily:"monospace" }}>{ev.conf}</div>
                            <div style={{ color:"#8ab0c8", fontFamily:"monospace" }}>{ev.score_A}</div>
                          </div>
                        );
                      })}
                    </>
                  )}
                </Card>
              </div>
            )
          )}

          {/* ── SETTINGS ── */}
          {tab==="settings" && onboardingDone && (
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
              <Card animate>
                <CardHdr title="⚙️ Engine Parameters — locked"/>
                {[
                  { k:"M4 LSTM-AE threshold", v:"0.110058",  note:"L1 — PERMANENTLY LOCKED. Do not retrain." },
                  { k:"XGBoost classes",       v:"24",        note:"Labels 0–23 (incl. Group E labels 22/23)" },
                  { k:"CUSUM alarm H",         v:"5.0",       note:"L3 — Label 21 gradual bearing wear" },
                  { k:"θ_initial",             v:healthData?.rolling_state?.theta_initial?.toFixed(6)||"1.881275", note:"L4 adaptive baseline" },
                  { k:"z_t buffer length",     v:"63",        note:"TCN-AE receptive field = 63 windows" },
                  { k:"Training basis",        v:"CIRA synthetic", note:"110 kW, 2980 RPM, 40 bar — single-pump v14.2" },
                  { k:"Sensor history buffer", v:"86,400 @ 1Hz = 24h", note:"v5.1 — RAM ring buffer, multi-client safe" },
                ].map((it,i) => (
                  <div key={i} style={{ paddingBottom:10, marginBottom:10,
                    borderBottom:"1px solid rgba(20,55,95,.3)" }}>
                    <div style={{ display:"flex", justifyContent:"space-between" }}>
                      <div style={{ fontSize:13, color:"#8ab0c8" }}>{it.k}</div>
                      <div style={{ fontSize:13, color:"#00d4ff", fontFamily:"monospace" }}>{it.v}</div>
                    </div>
                    <div style={{ fontSize:10, color:"#2a5060", marginTop:2 }}>{it.note}</div>
                  </div>
                ))}
              </Card>
              <Card animate>
                <CardHdr title="🔌 Server Status"/>
                {healthData ? (
                  <div>
                    {[
                      { k:"Status",         v:healthData.status,                                     c:"#00e676" },
                      { k:"Uptime",         v:healthData.uptime_seconds+"s" },
                      { k:"M4 LSTM-AE",     v:healthData.models_loaded?.m4_lstm_ae?"Loaded":"Error",
                        c:healthData.models_loaded?.m4_lstm_ae?"#00e676":"#ff2244" },
                      { k:"M8 TCN-AE",      v:healthData.models_loaded?.m8_tcn_ae?"Loaded":"Inline placeholder",
                        c:healthData.models_loaded?.m8_tcn_ae?"#00e676":"#ffcc00" },
                      { k:"M7 XGBoost",     v:healthData.models_loaded?.m7_xgboost?"Loaded":"Error",
                        c:healthData.models_loaded?.m7_xgboost?"#00e676":"#ff2244" },
                      { k:"z_t buffer",     v:`${healthData.zt_buffer_state?.buffer_fill||0}/${healthData.zt_buffer_state?.buffer_capacity||63}` },
                      { k:"Sensor history", v:historyBufferState ?
                        `${historyBufferState.buffer_fill}/${historyBufferState.buffer_capacity} (${historyBufferState.fill_pct}%)` : "—" },
                      { k:"Active channels",v:`${connectedCount}/8` },
                      { k:"Last update",    v:lastUpdate||"—" },
                    ].map((it,i) => (
                      <div key={i} style={{ display:"flex", justifyContent:"space-between",
                        padding:"7px 0", borderBottom:"1px solid rgba(20,55,95,.25)" }}>
                        <div style={{ fontSize:13, color:"#5a9ab0" }}>{it.k}</div>
                        <div style={{ fontSize:13, color:it.c||"#c0d8f0", fontFamily:"monospace" }}>{it.v}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize:12, color:"#2a5060" }}>Connecting to server…</div>
                )}
              </Card>
              <Card animate style={{ gridColumn:"1/3" }}>
                <CardHdr title="⚠️ System Scope & Disclaimer"/>
                <div style={{ fontSize:12, color:"#4a7a8a", lineHeight:1.9 }}>
                  PumpSmart v14.2 — single-pump monitoring.{" "}
                  <strong style={{ color:"#c0d8f0" }}>110 kW, 7-stage at 2980 RPM, 40 bar, 45 m³/h.</strong>{" "}
                  Advisory only. Not for autonomous control. Expected real-world F1: 0.65–0.85 (C-26).
                  <span style={{ cursor:"pointer", color:"#00b4d8", marginLeft:8 }}
                    onClick={() => setTab("guide")}>→ 📖 Full Guide</span>
                </div>
              </Card>
            </div>
          )}

          {/* ── GUIDE ── */}
          {tab==="guide" && (
            <Card animate>
              <CardHdr title="📖 Guide & Disclaimer"/>
              <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                {[
                  { t:"🏭 Applicability", c:"#00d4ff",
                    b:"Trained exclusively for 110 kW, 7-stage centrifugal pump at 2980 RPM, 40 bar, 45 m³/h. Any other specification is out of distribution — results will be unreliable." },
                  { t:"🔌 M12 local test mode", c:"#00e676",
                    b:`Step 1 — Start the FastAPI server (terminal 1):\n  uvicorn app.main:app --port 8000 --reload\n\nStep 2 — Acknowledge in the dashboard, then go to Sensor Plugin and mark the channels your test script will send as "Data active".\n\nStep 3 — Generate adversarial sequences (terminal 2):\n  python src/module_12a_adversarial_generator.py --mode smoke\n\nStep 4 — Run the M12 runner (terminal 2):\n  python src/module_12b_adversarial_runner.py --mode smoke\n\nThe runner POSTs normalised 50×8 windows to /api/anomaly_detect. The dashboard updates in real time. The runner will inject normal sequences, then various fault classes (bearing wear, cavitation, seal failure, compound chains, Label 21 gradual wear, masked faults, multi-sensor failures, etc.).\n\nStep 5 — Use Predictions tab to rate each prediction as Correct/Incorrect/Unsure after you observe the system response.\n\nMulti-tester: 5 testers can watch the dashboard simultaneously — all see the same server-side history timeline (v5.1).` },
                  { t:"🔌 Real SCADA/PLC connection", c:"#00b4d8",
                    b:`POST to /api/anomaly_detect with JSON:\n{\n  "window": [[50 rows × 8 cols of M3-normalised float values]],\n  "pump_id": "PUMP-0032",\n  "cluster": "steady_state"  // or startup, high_load, cooldown\n}\n\nChannel order (column index 0→7):\n  Mot.SV · Pmp.SV · Mot.TV · Pmp.PV · Temp.SV · Pres.SV · Pmp.TV · Mot.PV\n\nNormalisation: divide each channel by its M3 cluster mean (loaded from M3_normalization_config.json). Normal operation → values near 0–1. Fault onset → drift above 1.0.` },
                  { t:"🚦 Alert states explained", c:"#ffcc00",
                    b:"NORMAL: S_n < 2.0 AND score_A below θ_t. Continue monitoring.\nWATCH: CUSUM S_n ≥ 2.0. Sub-threshold drift — CUSUM accumulating. Event log only, no popup. Schedule inspection.\nWARN: score_A ≥ θ_t. Popup appears. Physical inspection required.\nDANGER: score_A ≥ 1.5×θ_t. Popup + urgent. Immediate investigation required." },
                  { t:"✅ Acknowledge vs Verdict — critical distinction", c:"#ff8800",
                    b:"ACKNOWLEDGE (alert popup button):\n→ Operational reset — clears CUSUM S_n, z_t buffer, rolling baseline\n→ Use AFTER physical inspection confirms fault handled\n→ Does NOT write to active-learning data store\n→ Does NOT reset sensor history (forensic retention)\n\nVERDICT (Predictions tab — Correct / Incorrect / Unsure):\n→ The ONLY active-learning write point\n→ Submit AFTER physical investigation as your professional judgment\n→ Builds the synthetic-to-real bridge for model improvement (C-26)" },
                  { t:"⚠️ Model limitations", c:"#ff4466",
                    b:"Real-world F1: 0.65–0.85 (trained on synthetic data — C-26). Label 21 gradual bearing wear: earliest detection ~Week 5. Group C masked faults: confidence significantly reduced when sensor is masking underlying fault. Single-pump only — cross-pump hydraulic interactions not modelled. Always verify predictions physically before any maintenance action." },
                ].map((item,i) => (
                  <div key={i} style={{ padding:"12px 16px", background:"rgba(4,12,26,.65)",
                    border:`1px solid ${item.c}22`, borderLeft:`3px solid ${item.c}`, borderRadius:10 }}>
                    <div style={{ fontSize:13, fontWeight:600, color:item.c, marginBottom:6 }}>{item.t}</div>
                    <pre style={{ fontSize:12, color:"#5a8aaa", lineHeight:1.75,
                      whiteSpace:"pre-wrap", fontFamily:"inherit", margin:0 }}>{item.b}</pre>
                  </div>
                ))}

                {!onboardingDone && (
                  <GuidAck onAccept={() => { setOnboardingDone(true); setTab("dashboard"); }}/>
                )}
              </div>
            </Card>
          )}

        </div>
      </div>

      {/* Alert popup */}
      {onboardingDone && showPopup && prediction &&
       (alertState==="WARN"||alertState==="DANGER") && (
        <AlertPopup prediction={prediction} alertState={alertState}
          onAcknowledge={handleAcknowledge} onDismiss={() => setShowPopup(false)}/>
      )}

      <style>{`
        @keyframes fadeSlide{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
        @keyframes fadeIn{from{opacity:0}to{opacity:1}}
        code{font-family:monospace}
      `}</style>
    </div>
  );
}

function GuidAck({ onAccept }) {
  const [checked, setChecked] = useState(false);
  return (
    <div style={{ padding:20, background:"rgba(0,180,220,.07)",
      border:"2px solid rgba(0,200,220,.3)", borderRadius:12 }}>
      <div style={{ fontSize:14, fontWeight:600, color:"#00d4ff", marginBottom:8 }}>☑️ Acknowledgement required</div>
      <label style={{ display:"flex", gap:12, cursor:"pointer", marginBottom:12 }}>
        <div onClick={() => setChecked(c=>!c)}
          style={{ width:22, height:22, borderRadius:6, flexShrink:0,
            border:`2px solid ${checked?"#00d4ff":"rgba(40,90,130,.6)"}`,
            background:checked?"rgba(0,200,220,.15)":"transparent",
            display:"flex", alignItems:"center", justifyContent:"center", cursor:"pointer" }}>
          {checked&&<span style={{ color:"#00d4ff", fontSize:14, fontWeight:700 }}>✓</span>}
        </div>
        <span style={{ fontSize:13, color:"#8ab0c8", lineHeight:1.6 }}>
          I confirm my pump matches 110 kW, 7-stage, 40 bar, 2980 RPM. All predictions are advisory only.
        </span>
      </label>
      <button onClick={()=>checked&&onAccept()} style={{ width:"100%", padding:11, borderRadius:9,
        cursor:checked?"pointer":"not-allowed", fontSize:14, fontWeight:700,
        background:checked?"rgba(0,180,220,.18)":"rgba(20,55,95,.2)",
        border:`1px solid ${checked?"rgba(0,200,220,.4)":"rgba(30,70,110,.3)"}`,
        color:checked?"#00d4ff":"#2a5060" }}>
        {checked?"✅ Acknowledged — enter dashboard →":"Tick the box above to continue"}
      </button>
    </div>
  );
}

window.PumpSmartApp = App;