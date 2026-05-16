// PumpSmart v14.2 — Industrial Dashboard Mockup
// Version 2.0 — May 2026
// Changes from v1.0:
//   v5.0-A: Popup Acknowledge = operational reset ONLY (no active-learning write)
//           Predictions tab Correct/Incorrect/Unsure = ONLY write point to active-learning data
//           Popup text clarified per ISA-18.2 alarm management
//   v5.0-B: 27-column active-learning schema defined in ACTIVE_LEARNING_SCHEMA constant
//   v5.0-C: HF Datasets API persistence pattern (push_learning_row) — not local disk
//   v5.0-D: Single-pump badge added to sidebar pump panel
//   Concern 3: Sensor Plugin now shows per-cluster ranges, active cluster highlighted
//   Concern 4: Risk gauge needle snaps immediately on state-change transitions (NUREG-0700)

import { useState, useEffect, useRef, useCallback } from "react";

const STATES = ["NORMAL","WATCH","WARN","DANGER"];
const SD = {
  NORMAL:{ needle:0.08,color:"#00e676",label:"NORMAL",sA:"0.048",sB:"-0.003",sC:"0.12",cu:"0.31",theta:"0.121",cf:"98.2",fault:"Normal operation",vibe:4.3,pres:39.8,temp:65,rpm:2980,
    diag:[{n:"Bearing wear",p:6,c:"#7c6fcd",icon:"⚙"},{n:"Cavitation",p:9,c:"#00b4d8",icon:"💧"},{n:"Seal leakage",p:4,c:"#00e676",icon:"🔩"},{n:"Overloading",p:3,c:"#00e676",icon:"⚡"}],
    alerts:[{icon:"✅",col:"#00e676",tag:"Normal",time:"08:00 AM",txt:"Pump started. Startup cluster detected. All 8 sensors live."}]},
  WATCH:{ needle:0.40,color:"#ffcc00",label:"WATCH",sA:"0.082",sB:"0.019",sC:"0.28",cu:"3.40",theta:"0.124",cf:"61.4",fault:"Bearing wear — gradual",vibe:5.1,pres:39.2,temp:67,rpm:2977,
    diag:[{n:"Bearing wear",p:38,c:"#ffcc00",icon:"⚙"},{n:"Cavitation",p:12,c:"#00b4d8",icon:"💧"},{n:"Seal leakage",p:8,c:"#00e676",icon:"🔩"},{n:"Overloading",p:5,c:"#00e676",icon:"⚡"}],
    alerts:[
      {icon:"👁",col:"#ffcc00",tag:"Bearing — gradual",time:"12:32 PM",txt:"CUSUM S_n crossed 2.0 — sub-threshold bearing drift accumulating over past 6 hours."},
      {icon:"✅",col:"#00e676",tag:"Normal",time:"08:00 AM",txt:"Pump started. All 8 sensors live."}]},
  WARN:{ needle:0.64,color:"#ff8800",label:"WARNING",sA:"0.138",sB:"0.031",sC:"0.54",cu:"2.14",theta:"0.121",cf:"82.7",fault:"Cavitation",vibe:5.8,pres:36.4,temp:71,rpm:2966,
    diag:[{n:"Bearing wear",p:14,c:"#7c6fcd",icon:"⚙"},{n:"Cavitation",p:83,c:"#ff8800",icon:"💧"},{n:"Seal leakage",p:22,c:"#00b4d8",icon:"🔩"},{n:"Overloading",p:8,c:"#00e676",icon:"⚡"}],
    alerts:[
      {icon:"⚠️",col:"#ff8800",tag:"Cavitation",time:"12:47 PM",txt:"Pump entered WARNING. Pressure oscillations on suction side. Check NPSHa ≥ NPSHr + 0.5 m."},
      {icon:"👁",col:"#ffcc00",tag:"Bearing — gradual",time:"12:32 PM",txt:"CUSUM S_n 2.14 — gradual drift accumulating."},
      {icon:"✅",col:"#00e676",tag:"Normal",time:"08:00 AM",txt:"Pump started. All 8 sensors live."}]},
  DANGER:{ needle:0.87,color:"#ff2244",label:"DANGER",sA:"0.241",sB:"0.048",sC:"1.82",cu:"6.80",theta:"0.121",cf:"77.3",fault:"Seal failure + Cavitation",vibe:7.4,pres:33.1,temp:79,rpm:2941,
    diag:[{n:"Bearing wear",p:31,c:"#ff8800",icon:"⚙"},{n:"Cavitation",p:91,c:"#ff2244",icon:"💧"},{n:"Seal leakage",p:74,c:"#ff2244",icon:"🔩"},{n:"Overloading",p:18,c:"#ffcc00",icon:"⚡"}],
    alerts:[
      {icon:"🔴",col:"#ff2244",tag:"Seal+Cavitation",time:"01:14 PM",txt:"DANGER: Compound fault — seal failure driving high-head cavitation. Reduce speed immediately."},
      {icon:"⚠️",col:"#ff8800",tag:"Cavitation",time:"12:47 PM",txt:"Cavitation WARNING preceded current DANGER state."},
      {icon:"👁",col:"#ffcc00",tag:"Bearing",time:"12:32 PM",txt:"CUSUM drift accumulation began here."},
      {icon:"✅",col:"#00e676",tag:"Normal",time:"08:00 AM",txt:"Pump started normally."}]},
};

// ── Per-cluster sensor ranges (v5.0 Concern 3 fix) ───────────────────────
// Each sensor has ranges for all 4 operating clusters.
// Startup vibration 2-4x steady-state = correct physics (shaft resonance, NOT a fault).
// Cooldown Pmp.TV may go below ambient by 1.4°C due to flash evaporation at 40 bar (C-09).
const CLUSTER_LABELS = ["Startup","Steady-state","High-load","Cooldown"];
const CLUSTER_COLORS = ["#ffbb00","#00e676","#ff8800","#00b4d8"];

const SENSORS = [
  {id:"mot_sv",name:"Motor Vibration (RMS)",unit:"mm/s",ch:"Mot.SV",
    clusterRanges:["7–14","3.5–5.5","4.5–6.5","2.5–4.5"],
    desc:"Broadband peak acceleration — motor bearing side",icon:"〰️",headroom:0.1333,gainP95:0.758,
    note:"Startup values 2–4× higher — shaft resonance, NOT a fault"},
  {id:"pmp_sv",name:"Pump Vibration (RMS)",unit:"mm/s",ch:"Pmp.SV",
    clusterRanges:["6–12","3.0–5.0","4.0–6.0","2.0–4.0"],
    desc:"Broadband peak acceleration — pump casing",icon:"〰️",headroom:0.1509,gainP95:0.652,
    note:"Startup values 2–4× higher — shaft resonance, NOT a fault"},
  {id:"mot_tv",name:"Motor Winding Temperature",unit:"°C",ch:"Mot.TV",
    clusterRanges:["65–80","60–75","70–82","58–72"],
    desc:"Stator winding temperature — thermal overload indicator",icon:"🌡️",headroom:0.1468,gainP95:0.028},
  {id:"pmp_pv",name:"Pump Discharge Pressure",unit:"bar",ch:"Pmp.PV",
    clusterRanges:["0–41","38–41","39–42","10–38"],
    desc:"Discharge side pressure — main operating parameter",icon:"🔵",headroom:0.1219,gainP95:0.939,
    note:"Ramps from 0 during startup — correct physics"},
  {id:"temp_sv",name:"Bearing Housing Temperature",unit:"°C",ch:"Temp.SV",
    clusterRanges:["60–75","55–70","65–78","50–65"],
    desc:"Bearing housing — early bearing fault indicator",icon:"🌡️",headroom:0.1524,gainP95:0.067},
  {id:"pres_sv",name:"Suction Side Pressure",unit:"bar",ch:"Pres.SV",
    clusterRanges:["0.5–3.5","1.5–3.5","1.0–3.0","0.3–2.5"],
    desc:"Suction pressure — NPSHa monitoring",icon:"🔵",headroom:0.1157,gainP95:0.883},
  {id:"pmp_tv",name:"Pump Casing Temperature",unit:"°C",ch:"Pmp.TV",
    clusterRanges:["55–70","50–65","60–72","35–55"],
    desc:"Pump body temperature — flash evaporation indicator",icon:"🌡️",headroom:0.1599,gainP95:0.034,
    note:"Cooldown may drop below ambient ≤1.4°C — flash evaporation physics (C-09), not a sensor fault"},
  {id:"mot_pv",name:"Motor Power Draw",unit:"kW",ch:"Mot.PV",
    clusterRanges:["60–115","100–115","108–120","30–100"],
    desc:"Electrical power input — overloading detection",icon:"⚡",headroom:0.2358,gainP95:0.979},
];
const HEADROOM_FLAG=0.10;
const HEADROOM_WARN=0.14;

// ── 27-column active-learning schema v1.0 (v5.0-B — LOCKED) ────────────
// Stored as reference; actual rows written via push_learning_row() to HF Datasets API.
// WRITE POINT: Predictions tab Correct/Incorrect/Unsure buttons ONLY.
// /api/acknowledge (popup) does NOT write here — operational reset only.
const ACTIVE_LEARNING_SCHEMA = {
  version: "1.0",
  write_point: "predictions_tab_verdict_buttons_only",
  persistence: "huggingface_datasets_api",
  repo: "pumpsmart-active-learning",
  columns: [
    "timestamp_utc","pump_id","prediction_id","cluster_id",
    "predicted_label_int","predicted_label_name","confidence_pct",
    "score_A","score_B","score_C","cusum_s_n","theta_t","alert_state",
    "m8p6_sensor_flag","m8p6_flagged_channels",
    "mahal_dist","ood_flag",
    "raw_sensor_window","top_3_shap_features",
    "operator_verdict","operator_correct_label",
    "verdict_timestamp_utc","time_to_verdict_seconds",
    "physical_inspection_done","inspection_notes",
    "data_source","consent_granted_by"
  ]
};

const HISTORY_EVENTS=[
  {date:"2026-05-13",time:"01:14 PM",state:"DANGER",fault:"Seal failure + Cavitation",conf:"77.3%",action:"Operator acknowledged — reducing flow rate",verified:"Pending"},
  {date:"2026-05-13",time:"12:47 PM",state:"WARN",fault:"Cavitation",conf:"82.7%",action:"Alert raised — suction valve checked",verified:"Confirmed ✓"},
  {date:"2026-05-13",time:"12:32 PM",state:"WATCH",fault:"Bearing wear — gradual",conf:"61.4%",action:"CUSUM S_n crossed 2.0 — monitoring",verified:"Confirmed ✓"},
  {date:"2026-05-12",time:"03:22 PM",state:"WARN",fault:"Overloading",conf:"79.1%",action:"Flow reduced — alert cleared in 18 min",verified:"Confirmed ✓"},
  {date:"2026-05-12",time:"09:15 AM",state:"NORMAL",fault:"—",conf:"98.5%",action:"Routine startup — all systems normal",verified:"N/A"},
  {date:"2026-05-11",time:"02:40 PM",state:"WATCH",fault:"Bearing wear — gradual",conf:"58.2%",action:"CUSUM drift noted — inspection scheduled",verified:"Confirmed ✓"},
  {date:"2026-05-10",time:"11:05 AM",state:"WARN",fault:"Sensor failure (Pres.SV)",conf:"71.8%",action:"Sensor cable reseated — signal restored",verified:"Confirmed ✓"},
];

const ONBOARDING_POINTS=[
  {icon:"⚙️",title:"Exact pump specification match required",text:"Trained exclusively for 110 kW, 7-stage centrifugal pump at 2980 RPM, 40 bar, 45 m³/h. Any other specification will produce unreliable results."},
  {icon:"🧠",title:"Advisory only — not autonomous control",text:"All predictions require human review. No automated shutdown or physical action should ever be triggered directly from model output alone."},
  {icon:"🔬",title:"Trained on physics-synthetic data",text:"Model was trained on CIRA-anchored physics-synthetic data. Confidence scores on real-world faults may be lower than on simulated training data (expected F1 = 0.65–0.85). Always verify predictions physically."},
  {icon:"👷",title:"Certified engineer verification mandatory",text:"Every WARN or DANGER prediction must be physically inspected by a qualified mechanical or instrumentation engineer before any maintenance action."},
  {icon:"📊",title:"Confidence threshold — ≥85% before acting",text:"Predictions below 70% are flagged UNKNOWN FAULT. Low confidence does not mean no fault — it means multiple fault types are plausible. Inspect regardless."},
  {icon:"🔌",title:"Sensor quality is your responsibility",text:"Calibration, signal conditioning, and wiring quality directly impact accuracy. The system cannot self-diagnose hardware sensor drift unless a reference channel remains valid."},
];

// ── Sparkline ─────────────────────────────────────────────────────────────
function Sparkline({color,anomaly,h=32}){
  const pts=useRef(Array.from({length:20},(_,i)=>{const b=50+Math.sin(i*.4)*8;return Math.min(90,Math.max(10,b+(Math.random()-.5)*5+(anomaly?i*.8:0)));})).current;
  const path=pts.map((y,i)=>`${i===0?"M":"L"}${(i/(pts.length-1))*100},${y}`).join(" ");
  return(<svg viewBox="0 0 100 100" style={{width:"100%",height:h,display:"block"}} preserveAspectRatio="none">
    <defs><linearGradient id={`sg${color.slice(1)}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity="0.35"/><stop offset="100%" stopColor={color} stopOpacity="0"/></linearGradient></defs>
    <path d={path+` L100,100 L0,100 Z`} fill={`url(#sg${color.slice(1)})`}/>
    <path d={path} fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>);
}

// ── ArcGauge ──────────────────────────────────────────────────────────────
function ArcGauge({pct,color,size=60}){
  const canv=useRef(); const prev=useRef(0); const anim=useRef();
  useEffect(()=>{
    const target=pct; let cur=prev.current;
    const draw=(p)=>{
      const c=canv.current; if(!c)return;
      const ctx=c.getContext("2d"); const cx=size/2,cy=size/2,r=size*.38;
      ctx.clearRect(0,0,size,size);
      const s=-Math.PI*.85,e=Math.PI*.85;
      ctx.beginPath(); ctx.arc(cx,cy,r,s,e); ctx.strokeStyle="rgba(255,255,255,.07)"; ctx.lineWidth=5; ctx.lineCap="round"; ctx.stroke();
      if(p>0){const ea=s+(e-s)*(p/100); ctx.beginPath(); ctx.arc(cx,cy,r,s,ea); ctx.strokeStyle=color; ctx.lineWidth=5; ctx.lineCap="round"; ctx.stroke();
        const tx=cx+Math.cos(ea)*r,ty=cy+Math.sin(ea)*r;
        ctx.beginPath(); ctx.arc(tx,ty,4,0,Math.PI*2); ctx.fillStyle=color; ctx.shadowColor=color; ctx.shadowBlur=8; ctx.fill(); ctx.shadowBlur=0;}
    };
    cancelAnimationFrame(anim.current);
    const step=()=>{cur+=(target-cur)*.12;draw(cur);if(Math.abs(cur-target)>.3){anim.current=requestAnimationFrame(step);}else{draw(target);prev.current=target;}};
    step(); return()=>cancelAnimationFrame(anim.current);
  },[pct,color,size]);
  return <canvas ref={canv} width={size} height={size}/>;
}

// ── RiskGauge — v5.0 Concern 4 fix ───────────────────────────────────────
// Needle SNAPS immediately on state-change transitions (NUREG-0700 safety-critical display).
// Smooth 0.08 interpolation only within the same state.
function RiskGauge({state}){
  const canv=useRef();
  const prevNeedle=useRef(0);
  const prevState=useRef(state);
  const animRef=useRef();
  const d=SD[state];

  useEffect(()=>{
    const target=d.needle;
    const stateChanged = prevState.current !== state;

    // If state changed: snap immediately to new position (NUREG-0700)
    // If within same state: smooth interpolation at 0.08 factor
    let cur = stateChanged ? target : prevNeedle.current;
    prevState.current = state;

    const draw=(n)=>{
      const c=canv.current; if(!c)return;
      const ctx=c.getContext("2d"); const W=240,H=160,cx=120,cy=150,R=105;
      ctx.clearRect(0,0,W,H);
      const bgG=ctx.createRadialGradient(cx,cy,50,cx,cy,R+25);
      bgG.addColorStop(0,"rgba(8,24,52,.9)"); bgG.addColorStop(1,"rgba(4,12,28,.1)");
      ctx.beginPath(); ctx.arc(cx,cy,R+20,-Math.PI,0); ctx.fillStyle=bgG; ctx.fill();
      [{c:"#00e676",s:Math.PI,e:Math.PI*1.28},{c:"#aacc00",s:Math.PI*1.28,e:Math.PI*1.54},{c:"#ffbb00",s:Math.PI*1.54,e:Math.PI*1.76},{c:"#ff6600",s:Math.PI*1.76,e:Math.PI*1.92},{c:"#ff2244",s:Math.PI*1.92,e:Math.PI*2}]
        .forEach(({c:col,s,e})=>{ctx.beginPath();ctx.arc(cx,cy,R,s,e);ctx.strokeStyle=col+"44";ctx.lineWidth=24;ctx.lineCap="butt";ctx.stroke();ctx.beginPath();ctx.arc(cx,cy,R,s,e);ctx.strokeStyle=col;ctx.lineWidth=14;ctx.stroke();});
      [{t:"Normal",a:Math.PI*1.14,c:"#00e676"},{t:"Watch",a:Math.PI*1.41,c:"#aacc00"},{t:"Warning",a:Math.PI*1.65,c:"#ffbb00"},{t:"Critical",a:Math.PI*1.96,c:"#ff2244"}]
        .forEach(({t,a,c:col})=>{const lx=cx+Math.cos(a)*(R-30),ly=cy+Math.sin(a)*(R-30);ctx.save();ctx.translate(lx,ly);ctx.rotate(a+Math.PI/2);ctx.font="bold 8.5px Inter,sans-serif";ctx.fillStyle=col;ctx.textAlign="center";ctx.fillText(t,0,0);ctx.restore();});
      const na=Math.PI+n*Math.PI, nx=cx+Math.cos(na)*(R-18), ny=cy+Math.sin(na)*(R-18);
      ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(nx,ny);ctx.strokeStyle=d.color+"55";ctx.lineWidth=10;ctx.lineCap="round";ctx.stroke();
      ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(nx,ny);ctx.strokeStyle=d.color;ctx.lineWidth=3;ctx.lineCap="round";ctx.stroke();
      ctx.beginPath();ctx.arc(nx,ny,5,0,Math.PI*2);ctx.fillStyle=d.color;ctx.shadowColor=d.color;ctx.shadowBlur=14;ctx.fill();ctx.shadowBlur=0;
      ctx.beginPath();ctx.arc(cx,cy,11,0,Math.PI*2);ctx.fillStyle="#060f1c";ctx.fill();
      ctx.beginPath();ctx.arc(cx,cy,11,0,Math.PI*2);ctx.strokeStyle=d.color+"88";ctx.lineWidth=2;ctx.stroke();
      ctx.beginPath();ctx.arc(cx,cy,5,0,Math.PI*2);ctx.fillStyle=d.color;ctx.fill();
      ctx.font="bold 20px Inter,sans-serif";ctx.fillStyle=d.color;ctx.textAlign="center";ctx.shadowColor=d.color;ctx.shadowBlur=16;ctx.fillText(d.label,cx,cy-32);ctx.shadowBlur=0;
      ctx.font="10px Inter,sans-serif";ctx.fillStyle="rgba(140,180,210,.65)";ctx.fillText("Predicted Failure Risk",cx,cy-16);
    };

    cancelAnimationFrame(animRef.current);

    if(stateChanged){
      // Immediate snap — draw target position right away, no animation
      draw(target);
      prevNeedle.current=target;
    } else {
      // Smooth within-state interpolation
      const step=()=>{cur+=(target-cur)*.08;draw(cur);if(Math.abs(cur-target)>.005){animRef.current=requestAnimationFrame(step);}else{draw(target);prevNeedle.current=target;}};
      step();
    }
    return()=>cancelAnimationFrame(animRef.current);
  },[state,d]);
  return <canvas ref={canv} width={240} height={160} style={{display:"block",margin:"0 auto"}}/>;
}

function Card({children,style={},animate=true}){
  const [vis,setVis]=useState(false);
  useEffect(()=>{const t=setTimeout(()=>setVis(true),50);return()=>clearTimeout(t);},[]);
  return(<div style={{background:"rgba(6,18,36,0.88)",border:"1px solid rgba(20,60,100,.5)",borderRadius:12,padding:16,backdropFilter:"blur(8px)",transition:"opacity .4s ease, transform .4s ease",opacity:animate?vis?1:0:1,transform:animate?vis?"translateY(0)":"translateY(14px)":"none",...style}}>{children}</div>);
}
function CardHdr({title,right}){
  return(<div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:14}}>
    <span style={{fontSize:11,color:"#5a9ab0",fontWeight:500,letterSpacing:.8,textTransform:"uppercase"}}>{title}</span>
    {right&&<div>{right}</div>}
  </div>);
}

const TAGCOLS={"Bearing — gradual":"#9b7fe8","Bearing":"#9b7fe8","Cavitation":"#00b4d8","Seal+Cavitation":"#ff2244","Seal leakage":"#ff2244","Overloading":"#ffbb00","Sensor":"#ccaa00","Normal":"#00e676"};
const STATECOLS={NORMAL:"#00e676",WATCH:"#ffcc00",WARN:"#ff8800",DANGER:"#ff2244"};

// ── BLOCKED GATE OVERLAY ─────────────────────────────────────────────────
function GateOverlay({onGoToGuide}){
  return(
    <div style={{position:"absolute",inset:0,background:"rgba(2,6,14,.85)",backdropFilter:"blur(12px)",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",zIndex:80,animation:"fadeIn .3s ease"}}>
      <div style={{textAlign:"center",maxWidth:420,padding:32}}>
        <div style={{fontSize:48,marginBottom:16}}>🔒</div>
        <div style={{fontSize:18,fontWeight:700,color:"#e0f0ff",marginBottom:10}}>Acknowledgement required</div>
        <div style={{fontSize:13,color:"#5a8aaa",lineHeight:1.7,marginBottom:24}}>
          You must read the guide and disclaimer and tick the acknowledgement box before accessing the monitoring system. This ensures you understand the pump specification requirements and advisory-only scope.
        </div>
        <button onClick={onGoToGuide} style={{padding:"12px 28px",borderRadius:10,background:"rgba(0,180,220,.18)",border:"1px solid rgba(0,200,220,.4)",color:"#00d4ff",cursor:"pointer",fontSize:14,fontWeight:700,letterSpacing:.3}}>
          📖 Go to Guide & Disclaimer →
        </button>
      </div>
    </div>
  );
}

// ── ONBOARDING POPUP ──────────────────────────────────────────────────────
function OnboardingPopup({onAccept,onViewGuide}){
  const [checked,setChecked]=useState(false);
  return(
    <div style={{position:"fixed",inset:0,background:"rgba(2,6,14,.9)",backdropFilter:"blur(16px)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:200,animation:"fadeIn .35s ease"}}>
      <div style={{background:"linear-gradient(160deg,rgba(8,18,36,.98),rgba(4,10,22,.99))",border:"1px solid rgba(0,180,220,.25)",borderRadius:16,padding:32,maxWidth:620,width:"92%",boxShadow:"0 32px 80px rgba(0,0,0,.7)",animation:"scaleIn .4s cubic-bezier(.16,1,.3,1)",maxHeight:"88vh",display:"flex",flexDirection:"column",overflow:"hidden"}}>
        <div style={{display:"flex",alignItems:"center",gap:14,marginBottom:20,flexShrink:0}}>
          <div style={{fontSize:36}}>⚙️</div>
          <div>
            <div style={{fontSize:18,fontWeight:700,color:"#e0f0ff",letterSpacing:.3}}>Before you proceed</div>
            <div style={{fontSize:12,color:"#3a6a7a",marginTop:3}}>PumpSmart Industrial Monitor — applicability acknowledgement</div>
          </div>
        </div>
        <div style={{background:"rgba(0,180,220,.07)",border:"1px solid rgba(0,180,220,.2)",borderRadius:10,padding:"12px 16px",marginBottom:16,flexShrink:0}}>
          <div style={{fontSize:11,color:"#3a6a7a",textTransform:"uppercase",letterSpacing:.8,marginBottom:8}}>🏭 This system is designed exclusively for</div>
          <div style={{display:"flex",gap:8,flexWrap:"wrap",marginBottom:10}}>
            {["⚡ 110 kW motor","🔧 7-stage centrifugal","📊 40 bar discharge","🌀 2980 RPM","💧 45 m³/h flow","📐 IEC 315 frame"].map((s,i)=>(
              <span key={i} style={{fontSize:11,padding:"3px 10px",borderRadius:6,background:"rgba(0,200,220,.1)",border:"1px solid rgba(0,200,220,.25)",color:"#00d4ff",fontWeight:600}}>{s}</span>
            ))}
          </div>
          <div style={{fontSize:12,color:"#ff8800",display:"flex",gap:6,alignItems:"flex-start"}}>
            <span>⚠️</span><span>Using this model on any other pump specification will produce unreliable predictions. Do not proceed if your pump does not match all parameters above.</span>
          </div>
        </div>
        <div style={{overflowY:"auto",flex:1,marginBottom:14,paddingRight:4}}>
          <div style={{fontSize:11,color:"#3a6a7a",textTransform:"uppercase",letterSpacing:.8,marginBottom:10}}>📋 Key points to understand before using this system</div>
          <div style={{display:"flex",flexDirection:"column",gap:8}}>
            {ONBOARDING_POINTS.map((pt,i)=>(
              <div key={i} style={{display:"flex",gap:12,padding:"10px 14px",background:"rgba(4,12,26,.6)",border:"1px solid rgba(20,55,95,.4)",borderRadius:10,animation:`fadeSlide .4s ${i*.05}s both`}}>
                <span style={{fontSize:20,flexShrink:0}}>{pt.icon}</span>
                <div>
                  <div style={{fontSize:12,fontWeight:600,color:"#c0d8f0",marginBottom:3}}>{pt.title}</div>
                  <div style={{fontSize:12,color:"#5a8aaa",lineHeight:1.5}}>{pt.text}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div style={{borderTop:"1px solid rgba(20,55,95,.5)",paddingTop:14,flexShrink:0}}>
          <label style={{display:"flex",alignItems:"flex-start",gap:12,cursor:"pointer",marginBottom:12}}>
            <div onClick={()=>setChecked(c=>!c)} style={{width:22,height:22,borderRadius:6,border:`2px solid ${checked?"#00d4ff":"rgba(40,90,130,.6)"}`,background:checked?"rgba(0,200,220,.15)":"transparent",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,marginTop:1,transition:"all .2s",cursor:"pointer"}}>
              {checked&&<span style={{color:"#00d4ff",fontSize:14,fontWeight:700}}>✓</span>}
            </div>
            <span style={{fontSize:13,color:"#8ab0c8",lineHeight:1.6}}>I confirm my pump matches the specifications above. I understand all predictions are <strong style={{color:"#c0d8f0"}}>advisory only</strong> and must be verified physically by a qualified engineer before any action is taken.</span>
          </label>
          <div style={{display:"flex",gap:10}}>
            <button onClick={onViewGuide} style={{flex:1,padding:"10px",borderRadius:9,background:"rgba(20,55,95,.3)",border:"1px solid rgba(40,90,130,.4)",color:"#5a9ab0",cursor:"pointer",fontSize:12,fontWeight:500}}>
              📖 Read full guide & disclaimer
            </button>
            <button onClick={()=>checked&&onAccept()} style={{flex:2,padding:"11px",borderRadius:9,background:checked?"rgba(0,180,220,.18)":"rgba(20,55,95,.2)",border:`1px solid ${checked?"rgba(0,200,220,.4)":"rgba(30,70,110,.3)"}`,color:checked?"#00d4ff":"#2a5060",cursor:checked?"pointer":"not-allowed",fontSize:14,fontWeight:700,transition:"all .25s"}}>
              {checked?"✓ I acknowledge — enter dashboard":"☑️ Tick the box above to continue"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── GUIDE TAB ─────────────────────────────────────────────────────────────
function GuideTab({onNavigate, onboardingDone, onAcceptFromGuide}){
  const [sec,setSec]=useState("applicability");
  const [checked,setChecked]=useState(false);
  const sections=[
    {id:"applicability",label:"🏭 Applicability Domain"},
    {id:"howto",label:"🚀 How to Use"},
    {id:"alerts",label:"🚨 Understanding Alerts"},
    {id:"limitations",label:"⚠️ Limitations & Assumptions"},
    {id:"faq",label:"❓ FAQs"},
    {id:"future",label:"🔭 Future Roadmap"},
  ];
  const P=({children,style={}})=><p style={{fontSize:13,color:"#8ab0c8",lineHeight:1.75,marginBottom:10,...style}}>{children}</p>;
  const B=({children})=><strong style={{color:"#c0d8f0"}}>{children}</strong>;
  const Sec=({children})=><div style={{fontSize:11,color:"#3a6a7a",textTransform:"uppercase",letterSpacing:.8,fontWeight:600,marginBottom:10,marginTop:18}}>{children}</div>;
  const Warn=({children})=><div style={{padding:"10px 14px",background:"rgba(255,136,0,.07)",border:"1px solid rgba(255,136,0,.2)",borderRadius:9,fontSize:12,color:"#ff8800",lineHeight:1.6,marginBottom:12,display:"flex",gap:8}}><span>⚠️</span><span>{children}</span></div>;
  const Good=({children})=><div style={{padding:"10px 14px",background:"rgba(0,200,100,.06)",border:"1px solid rgba(0,200,100,.2)",borderRadius:9,fontSize:12,color:"#00e676",lineHeight:1.6,marginBottom:12,display:"flex",gap:8}}><span>✅</span><span>{children}</span></div>;
  return(
    <div style={{display:"flex",gap:16,animation:"fadeSlide .4s ease"}}>
      <div style={{width:178,flexShrink:0}}>
        <div style={{background:"rgba(6,18,36,.88)",border:"1px solid rgba(20,60,100,.5)",borderRadius:12,padding:12,position:"sticky",top:0}}>
          <div style={{fontSize:10,color:"#2a5060",textTransform:"uppercase",letterSpacing:.8,marginBottom:10,padding:"0 4px"}}>📚 Contents</div>
          {sections.map(s=>(
            <div key={s.id} onClick={()=>setSec(s.id)} style={{padding:"9px 12px",borderRadius:8,cursor:"pointer",fontSize:12,color:sec===s.id?"#00d4ff":"#4a7a8a",background:sec===s.id?"rgba(0,180,220,.08)":"transparent",borderLeft:`3px solid ${sec===s.id?"#00d4ff":"transparent"}`,transition:"all .2s",marginBottom:2,fontWeight:sec===s.id?600:400,lineHeight:1.4}}>{s.label}</div>
          ))}
          {onboardingDone&&(
            <div style={{marginTop:14,padding:"10px 12px",background:"rgba(4,12,26,.6)",borderRadius:8,border:"1px solid rgba(20,55,95,.4)"}}>
              <div style={{fontSize:10,color:"#2a5060",marginBottom:6}}>🔗 Quick links</div>
              {[["🔌 Sensor Plugin","sensor"],["📈 Analytics","analytics"],["🧠 Predictions","predictions"]].map(([l,t])=>(
                <div key={t} onClick={()=>onNavigate(t)} style={{fontSize:11,color:"#00b4d8",cursor:"pointer",padding:"4px 0"}}>{l}</div>
              ))}
            </div>
          )}
        </div>
      </div>
      <div style={{flex:1,overflowY:"auto"}}>
        <Card animate>
          {sec==="applicability"&&<>
            <CardHdr title="🏭 Applicability Domain"/>
            <div style={{display:"flex",gap:8,flexWrap:"wrap",marginBottom:14}}>
              {[["⚡ Motor power","110 kW"],["🔧 Stages","7-stage"],["📊 Discharge","40 bar"],["🌀 Speed","2980 RPM"],["💧 Flow","45 m³/h"],["📏 Head","450 m"],["📐 Frame","IEC 315"],["🌊 Fluid","Water / similar"]].map(([k,v])=>(
                <div key={k} style={{background:"rgba(4,12,26,.7)",border:"1px solid rgba(20,55,95,.5)",borderRadius:8,padding:"8px 12px",minWidth:90}}>
                  <div style={{fontSize:10,color:"#2a5060"}}>{k}</div><div style={{fontSize:14,fontWeight:600,color:"#00d4ff",marginTop:3}}>{v}</div>
                </div>
              ))}
            </div>
            <Warn>If your pump does not match ALL parameters above, this model is operating outside its training distribution. Do not use on differently sized, single-stage, or non-centrifugal pumps.</Warn>
            <Sec>22 fault classes across 5 groups</Sec>
            {[{g:"🟢 Group A — Basic single faults",c:"#00e676",f:"Normal, Bearing wear, Impeller imbalance, Cavitation, Seal failure, Overloading, Sensor failure"},
              {g:"🔵 Group B — Compound fault chains",c:"#00b4d8",f:"Bearing+overloading, Cavitation+seal, Imbalance+bearing, Seal+cavitation (high-head), Overloading+bearing, Imbalance+cavitation"},
              {g:"🟡 Group C — Masked faults",c:"#ffbb00",f:"Bearing with vibration sensor masked, Cavitation with pressure sensor masked, Seal with drifting sensor, Overloading with stuck temp sensor"},
              {g:"🟠 Group D — Cyclic/Intermittent",c:"#ff8800",f:"Intermittent cavitation, Fast seal failure, Cyclic overloading"},
              {g:"🟣 Group E — Gradual wear",c:"#9b7fe8",f:"Very gradual bearing wear (1000-step drift, CUSUM-only detection — earliest detection ~Week 5)"},
            ].map((it,i)=>(
              <div key={i} style={{padding:"10px 14px",background:"rgba(4,12,26,.6)",border:`1px solid ${it.c}22`,borderLeft:`3px solid ${it.c}`,borderRadius:9,marginBottom:8}}>
                <div style={{fontSize:12,fontWeight:600,color:it.c,marginBottom:4}}>{it.g}</div>
                <div style={{fontSize:12,color:"#5a8aaa"}}>{it.f}</div>
              </div>
            ))}
            <Warn>Cross-pump effects not modelled. Multiple pumps on a shared manifold — hydraulic interactions are not captured. Single-pump monitoring only (v14.2).</Warn>
          </>}
          {sec==="howto"&&<>
            <CardHdr title="🚀 How to Use — Step by Step"/>
            <Sec>Step 1️⃣ — Day 1 setup (one-time only)</Sec>
            <P>Go to <B>🔌 Sensor Plugin</B>. Map each physical cable to its channel. Verify all 8 green dots. This is the only setup step — fully automatic after this.</P>
            <Good>After configuration the system runs completely automatically. You do not select operating modes, enter setpoints, or touch any controls during normal operation.</Good>
            <Sec>Step 2️⃣ — Normal monitoring</Sec>
            <P>The <B>🏠 Dashboard</B> updates automatically every ~50 seconds. Healthy steady-state: gauge at NORMAL 🟢, all vital badges green, CUSUM S_n near zero.</P>
            <Sec>Step 3️⃣ — WATCH state 👁</Sec>
            <P>CUSUM S_n ≥ 2.0. Sub-threshold drift accumulating — likely very early gradual bearing wear. Shown in event log only — no popup. Schedule bearing inspection at next planned maintenance. Do not reset — let CUSUM continue.</P>
            <Sec>Step 4️⃣ — WARNING ⚠️ or DANGER 🔴</Sec>
            <P>Popup appears automatically. Read the diagnosis, then choose:</P>
            {[{l:"✅ Acknowledge — action taken",c:"#00b4d8",d:"Resets CUSUM, z_t buffer, and rolling baseline so the alert state machine can proceed. Use AFTER physical inspection. This is an operational reset — not a verdict on the model's accuracy."},
              {l:"❌ Dismiss",c:"#ff4466",d:"Closes popup temporarily without resetting. Alert remains active in log and dashboard."}].map((b,i)=>(
              <div key={i} style={{padding:"9px 12px",background:"rgba(4,12,26,.6)",border:`1px solid ${b.c}30`,borderRadius:9,marginBottom:8}}>
                <div style={{fontSize:12,fontWeight:600,color:b.c,marginBottom:3}}>{b.l}</div>
                <div style={{fontSize:12,color:"#5a8aaa"}}>{b.d}</div>
              </div>
            ))}
            <Sec>Step 5️⃣ — Verify predictions (🧠 Predictions tab)</Sec>
            <P>After any alert, go to <B>🧠 Predictions</B> and click <B>Correct / Incorrect / Unsure</B> after physically investigating. This is where your professional judgment about the model's accuracy is recorded. This is separate from the Acknowledge button — Acknowledge is a control-room action; prediction verification is a calm-state action after investigation.</P>
            <Good>Your verified predictions are stored in the active-learning data repository and contribute to future model improvement. The more operators verify, the faster the synthetic-to-real gap narrows.</Good>
          </>}
          {sec==="alerts"&&<>
            <CardHdr title="🚨 The 4-Layer Detection Stack"/>
            {[{l:"L1 🔵 LSTM-AE (Baseline)",c:"#00b4d8",t:"Detects acute abrupt faults. Computes reconstruction error score_A per 50-step window. Threshold fixed at 0.110058 — never changes."},
              {l:"L2 🟣 TCN-AE (Compound chains)",c:"#9b7fe8",t:"Sequences of z_t vectors from L1 encoder. Detects multi-window drift (score_B) and fault chain transitions (score_C). Targets compound faults across hundreds of steps."},
              {l:"L3 🟡 CUSUM (Gradual wear)",c:"#ffcc00",t:"Accumulates score_B drift over weeks. Only reliable detector for Label 21 gradual bearing wear. Alarm H=5.0. Resets only on confirmed maintenance."},
              {l:"L4 🟢 Adaptive Threshold (False alarm control)",c:"#00e676",t:"Rolling 6-hour mean ± 3σ of score_A. θ_t = μ + 3σ. Crosspoint guard: θ_t > 1.5× initial → locks and raises DRIFT ALERT."},
            ].map((it,i)=>(
              <div key={i} style={{padding:"11px 14px",background:"rgba(4,12,26,.65)",border:`1px solid ${it.c}25`,borderLeft:`3px solid ${it.c}`,borderRadius:10,marginBottom:10}}>
                <div style={{fontSize:12,fontWeight:700,color:it.c,marginBottom:5}}>{it.l}</div>
                <div style={{fontSize:12,color:"#5a8aaa",lineHeight:1.6}}>{it.t}</div>
              </div>
            ))}
            <Sec>🚦 4 alert states</Sec>
            {[{s:"🟢 NORMAL",c:"#00e676",d:"score_A below threshold AND CUSUM < 2.0. Continue routine monitoring."},
              {s:"👁 WATCH",c:"#ffcc00",d:"CUSUM S_n ≥ 2.0. Sub-threshold drift — monitor closely, schedule inspection. Event log only — no popup."},
              {s:"⚠️ WARNING",c:"#ff8800",d:"score_A ≥ θ_t. Fault confirmed above baseline. Popup appears. Physical inspection required."},
              {s:"🔴 DANGER",c:"#ff2244",d:"score_A ≥ 1.5× θ_t. Significantly elevated. Popup appears. Immediate investigation required."},
            ].map((it,i)=>(
              <div key={i} style={{display:"flex",gap:10,padding:"9px 12px",background:`${it.c}08`,border:`1px solid ${it.c}25`,borderRadius:9,marginBottom:7,alignItems:"flex-start"}}>
                <span style={{fontSize:12,padding:"2px 9px",borderRadius:6,background:`${it.c}18`,color:it.c,fontWeight:700,flexShrink:0,whiteSpace:"nowrap"}}>{it.s}</span>
                <div style={{fontSize:12,color:"#8ab0c8",lineHeight:1.6}}>{it.d}</div>
              </div>
            ))}
          </>}
          {sec==="limitations"&&<>
            <CardHdr title="⚠️ Limitations, Assumptions & Honest Boundaries"/>
            <Warn>Reading this section fully before relying on predictions for any maintenance decision is strongly recommended.</Warn>
            <Sec>🔬 Training data basis</Sec>
            <P>Trained on <B>CIRA SACIP dataset anchor + physics-synthetic fault sequences</B>. Has never seen real-world fault data from a physical pump in production. Expected F1 on real faults = 0.65–0.85 (C-26). Always weight physical inspection equally with model output.</P>
            <Sec>📋 Key assumptions</Sec>
            {["🌊 Fluid is water or similar density (≈1000 kg/m³). Viscous or slurry fluids will shift sensor baselines unpredictably.",
              "🏞️ Pump draws from a large stable reservoir. Suction-side fluctuations from upstream processes are not modelled.",
              "❄️ Flash evaporation physics correctly handled — pump casing temperature BELOW ambient by up to 1.4°C during shutdown is expected and not flagged as sensor failure (C-09).",
              "⏱️ Sensor sampling rate ~1 Hz, 50-step windows every ~50 seconds. Significantly different rates require recalibration.",
              "🔧 After deployment on a new pump, 48-hour commissioning period required to validate cluster assignments.",
              "🔂 Single pump monitoring only (v14.2) — cross-pump hydraulic interactions not captured.",
            ].map((t,i)=>(<div key={i} style={{display:"flex",gap:8,padding:"8px 12px",background:"rgba(4,12,26,.6)",border:"1px solid rgba(20,55,95,.4)",borderRadius:8,fontSize:12,color:"#5a8aaa",lineHeight:1.55,marginBottom:6}}><span style={{flexShrink:0}}>→</span><span>{t}</span></div>))}
            <Sec>🚧 Known detection limits</Sec>
            {[{k:"⏳ Label 21 — Gradual bearing wear",v:"Earliest reliable detection ~Week 5. Do not use CUSUM S_n absence as confirmation of bearing health."},
              {k:"🎭 Group C — Masked faults",v:"When a sensor is masking an underlying fault, confidence is significantly reduced. Physical inspection of masked channel is mandatory."},
              {k:"⛓️ Compound chains (Group B)",v:"Model identifies the chain but causal direction must be confirmed physically. Treating secondary fault as primary leads to incorrect maintenance action."},
            ].map((it,i)=>(<div key={i} style={{padding:"10px 14px",background:"rgba(4,12,26,.6)",border:"1px solid rgba(20,55,95,.4)",borderLeft:"3px solid #ff8800",borderRadius:9,marginBottom:8}}>
              <div style={{fontSize:12,fontWeight:600,color:"#ff8800",marginBottom:4}}>{it.k}</div>
              <div style={{fontSize:12,color:"#5a8aaa",lineHeight:1.6}}>{it.v}</div>
            </div>))}
            <Good>The model is reliably good at detecting acute abrupt faults — cavitation, seal failure, overloading, imbalance — in steady-state operation with confidence ≥80% when the fault is fully developed and sensor channels are clean.</Good>
            <div style={{marginTop:14}}>
              <Sec>C-28 — Sensor ceiling-approach (M8p6)</Sec>
              <P>A fourth sensor failure mode is detected: sensors approaching their ISA-37 ceiling before flatline. Two CIRA channels are borderline by design:</P>
              <div style={{display:"flex",flexDirection:"column",gap:8,marginBottom:10}}>
                {[{k:"Suction Side Pressure (Pres.SV)",v:"Headroom 11.6% — tight by design: high-load ceiling 2.0× (C-18).",c:"#e67e22"},{k:"Discharge Pressure (Pmp.PV)",v:"Headroom 12.2% — startup ceiling 3.2× (C-17 ISO 13373-3).",c:"#ffcc00"}].map((it,i)=>(
                  <div key={i} style={{padding:"8px 12px",background:"rgba(4,12,26,.6)",border:`1px solid ${it.c}33`,borderLeft:`3px solid ${it.c}`,borderRadius:8}}>
                    <div style={{fontSize:12,fontWeight:600,color:it.c,marginBottom:3}}>{it.k}</div>
                    <div style={{fontSize:12,color:"#5a8aaa"}}>{it.v}</div>
                  </div>
                ))}
              </div>
              <P>In production on non-CIRA pumps, these channels will trigger an amber Field 6 calibration warning. Never alters fault label or confidence (Principle 14).</P>
            </div>
          </>}
          {sec==="faq"&&<>
            <CardHdr title="❓ Frequently Asked Questions"/>
            {[{q:"🔊 Model shows NORMAL but I hear unusual noise. What do I do?",a:"Trust your physical senses. The model works on 8 specific sensor statistical patterns — it cannot hear acoustic emission, detect oil discolouration, or feel shaft play. Physical inspection always takes priority."},
              {q:"📉 Why is confidence only 61% for a WATCH alert?",a:"WATCH is driven by the CUSUM accumulator (L3), not XGBoost (M7). Low confidence correctly indicates the fault has not yet fully manifested — expected for Label 21 gradual bearing wear at early stages."},
              {q:"😮 DANGER popup appeared but the pump seems fine. Can I dismiss it?",a:"Yes, but do not ignore it. DANGER means score_A crossed 1.5× the adaptive threshold. Physically inspect before dismissing. If no fault is found, the trigger may have been a process transient. Acknowledge after inspection."},
              {q:"🔧 We replaced the bearing — should I click Acknowledge?",a:"Yes. Always click Acknowledge after confirmed maintenance. This resets CUSUM accumulator, z_t buffer, and rolling baseline — essential for re-establishing a clean normal baseline after repair. Then go to the Predictions tab to rate the prediction accuracy."},
              {q:"📐 Can I use this on a 90 kW or 132 kW pump?",a:"No. Normalization baselines, fault signatures, and physics-synthetic training data are specific to 110 kW, 7-stage, 40 bar. A different pump will have different vibration envelopes and thermal time constants."},
              {q:"🔌 What happens if one sensor fails during operation?",a:"Go to Sensor Plugin — the affected channel shows a red dot. The model partially pauses on that channel. Monitoring continues on remaining channels. Do not rely on the model after more than 2 simultaneous sensor failures."},
            ].map((item,i)=>(
              <div key={i} style={{marginBottom:12,padding:"14px 16px",background:"rgba(4,12,26,.65)",border:"1px solid rgba(20,55,95,.45)",borderRadius:10}}>
                <div style={{fontSize:13,fontWeight:600,color:"#c0d8f0",marginBottom:8,lineHeight:1.5}}>{item.q}</div>
                <div style={{fontSize:12,color:"#5a8aaa",lineHeight:1.7}}>A: {item.a}</div>
              </div>
            ))}
          </>}
          {sec==="future"&&<>
            <CardHdr title="🔭 Future Roadmap & Planned Upgrades"/>
            {[{phase:"🟢 Near-term (v15.0)",c:"#00e676",items:["🔄 Multi-pump support — 2–5 pumps in parallel with cross-pump interaction modelling","📱 Mobile-responsive dashboard for field tablet use","🔗 REST API client library for direct SCADA and DCS integration","🎓 Real sensor data fine-tuning loop — active learning from operator-verified predictions"]},
              {phase:"🔵 Medium-term (v16.0)",c:"#00b4d8",items:["⏳ Remaining useful life (RUL) estimation for bearings from CUSUM trajectory","🌀 Variable speed drive (VSD) support — extended normalization for variable-RPM operation","📄 PDF report export — auto-generated maintenance reports with 7-field output","⚙️ Additional pump classes — 55 kW 5-stage and 160 kW 9-stage via transfer learning"]},
              {phase:"🟣 Long-term (v17.0+)",c:"#9b7fe8",items:["🏭 Digital twin integration — physics-simulation overlay on ML anomaly detection","📡 Vibration spectral analysis — FFT-based frequency domain fault signatures","☁️ Cloud-based fleet monitoring — anonymised aggregated learning across deployments","🛡️ IEC 61511 functional safety assessment for SIL-rated advisory systems"]},
            ].map((phase,i)=>(
              <div key={i} style={{background:"rgba(4,12,26,.65)",border:`1px solid ${phase.c}25`,borderLeft:`3px solid ${phase.c}`,borderRadius:10,padding:"14px 16px",marginBottom:12}}>
                <div style={{fontSize:13,fontWeight:600,color:phase.c,marginBottom:10}}>{phase.phase}</div>
                {phase.items.map((item,j)=>(<div key={j} style={{display:"flex",gap:8,fontSize:12,color:"#8ab0c8",marginBottom:5}}><span style={{flexShrink:0}}>→</span><span>{item}</span></div>))}
              </div>
            ))}
          </>}
        </Card>

        {/* Acknowledgement box — shown in guide tab when not yet acknowledged */}
        {!onboardingDone&&(
          <div style={{marginTop:16,padding:20,background:"rgba(0,180,220,.07)",border:"2px solid rgba(0,200,220,.3)",borderRadius:12,animation:"fadeSlide .4s ease"}}>
            <div style={{fontSize:14,fontWeight:600,color:"#00d4ff",marginBottom:8}}>☑️ Acknowledgement required to proceed</div>
            <div style={{fontSize:13,color:"#5a8aaa",lineHeight:1.6,marginBottom:14}}>You have now read the Guide & Disclaimer. Please tick the box below to confirm you understand the applicability domain and advisory-only scope of this system, then click the button to enter the monitoring dashboard.</div>
            <label style={{display:"flex",alignItems:"flex-start",gap:12,cursor:"pointer",marginBottom:14}}>
              <div onClick={()=>setChecked(c=>!c)} style={{width:22,height:22,borderRadius:6,border:`2px solid ${checked?"#00d4ff":"rgba(40,90,130,.6)"}`,background:checked?"rgba(0,200,220,.15)":"transparent",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,marginTop:1,transition:"all .2s",cursor:"pointer"}}>
                {checked&&<span style={{color:"#00d4ff",fontSize:14,fontWeight:700}}>✓</span>}
              </div>
              <span style={{fontSize:13,color:"#8ab0c8",lineHeight:1.6}}>I have read the guide and disclaimer. I confirm my pump matches the 110 kW, 7-stage, 40 bar, 2980 RPM specification. I understand all predictions are <strong style={{color:"#c0d8f0"}}>advisory only</strong> and must be verified physically by a qualified engineer before any action is taken.</span>
            </label>
            <button onClick={()=>checked&&onAcceptFromGuide()} style={{width:"100%",padding:"12px",borderRadius:9,background:checked?"rgba(0,180,220,.18)":"rgba(20,55,95,.2)",border:`1px solid ${checked?"rgba(0,200,220,.4)":"rgba(30,70,110,.3)"}`,color:checked?"#00d4ff":"#2a5060",cursor:checked?"pointer":"not-allowed",fontSize:14,fontWeight:700,transition:"all .25s",letterSpacing:.3}}>
              {checked?"✅ Acknowledged — proceed to dashboard →":"Tick the box above to unlock the dashboard"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═════════════════════════════════════════════════════════════════════════
export default function App(){
  const [state,setState]=useState("WARN");
  const [tab,setTab]=useState("dashboard");
  const [popupDismissed,setPopupDismissed]=useState(false);
  const [onboardingDone,setOnboardingDone]=useState(false);
  const [showOnboardingPopup,setShowOnboardingPopup]=useState(true);
  const [sensorConnected,setSensorConnected]=useState({mot_sv:true,pmp_sv:true,mot_tv:true,pmp_pv:true,temp_sv:true,pres_sv:false,pmp_tv:true,mot_pv:true});
  // v5.0 Concern 3: track active cluster for per-cluster range display
  const [activeCluster,setActiveCluster]=useState(1); // 0=Startup 1=Steady-state 2=High-load 3=Cooldown
  // v5.0-A: track prediction verdict state (Predictions tab — ONLY active-learning write point)
  const [verdictSent,setVerdictSent]=useState(null);
  const trendRef=useRef(); const cusumRef=useRef(); const analyticsRef=useRef();
  const trendChart=useRef(null); const cusumChart=useRef(null); const analyticsChart=useRef(null);
  const d=SD[state];

  useEffect(()=>{setPopupDismissed(false);},[state]);

  const GATED_TABS=["dashboard","sensor","analytics","predictions","history","settings"];

  const handleTabClick=(tabId)=>{
    if(!onboardingDone && GATED_TABS.includes(tabId)){
      setTab("guide_gate");
      return;
    }
    // Reset verdict display on tab change
    if(tabId==="predictions") setVerdictSent(null);
    setTab(tabId);
  };

  const buildChart=(canvasRef,chartRef,config)=>{
    if(!canvasRef.current)return;
    if(chartRef.current){chartRef.current.destroy();chartRef.current=null;}
    const C=window.Chart; if(!C)return;
    chartRef.current=new C(canvasRef.current,config);
  };
  const mkLabels=(n,step=15)=>Array.from({length:n},(_,i)=>i%Math.ceil(n/8)===0?`${11+Math.floor(i*step/60)}:${String((i*step)%60).padStart(2,"0")}`:"");
  const mkData=(base,amp,drift,n=24)=>Array.from({length:n},(_,i)=>+(base+Math.sin(i*.5)*amp+(Math.random()-.5)*amp*.5+(["WARN","DANGER"].includes(state)?drift*i:0)).toFixed(2));
  const chartDefaults={responsive:true,maintainAspectRatio:false,animation:{duration:600},plugins:{legend:{display:false},tooltip:{mode:"index",intersect:false,backgroundColor:"#0a1628",borderColor:"#1a3a5a",borderWidth:1,titleColor:"#6ab0cc",bodyColor:"#c0d8f0",padding:10}},scales:{x:{ticks:{color:"#2a5060",font:{size:10}},grid:{color:"rgba(20,50,80,.4)"},border:{color:"rgba(20,50,80,.3)"}},y:{ticks:{color:"#2a5060",font:{size:10}},grid:{color:"rgba(20,50,80,.4)"},border:{color:"rgba(20,50,80,.3)"}}}};

  const buildTrend=useCallback(()=>buildChart(trendRef,trendChart,{type:"line",data:{labels:mkLabels(24),datasets:[
    {label:"Discharge pressure (bar)",data:mkData(39,1.2,-.08),borderColor:"#00b4d8",borderWidth:2,pointRadius:0,tension:.4,fill:false},
    {label:"Winding temp (°C)",data:mkData(67,2,.1),borderColor:"#ff8800",borderWidth:2,pointRadius:0,tension:.4,fill:false,yAxisID:"y2"},
    {label:"Motor vibration (mm/s)",data:mkData(4.5,.4,.04),borderColor:"#00e676",borderWidth:2,pointRadius:0,tension:.4,fill:false,yAxisID:"y3"},
    {label:"Steady-state baseline",data:Array(24).fill(39),borderColor:"rgba(100,160,200,.2)",borderWidth:1,borderDash:[6,4],pointRadius:0,fill:false},
  ]},options:{...chartDefaults,scales:{...chartDefaults.scales,y2:{display:false},y3:{display:false}}}}),[state]);

  const buildCusum=useCallback(()=>{
    const target=parseFloat(d.cu); const lineCol=state==="DANGER"?"#ff2244":state==="WARN"?"#ff8800":state==="WATCH"?"#ffcc00":"#00e676";
    buildChart(cusumRef,cusumChart,{type:"line",data:{labels:Array.from({length:30},(_,i)=>i%10===0?`-${29-i}m`:""),datasets:[
      {data:Array.from({length:30},(_,i)=>+Math.max(0,target*(i/29)+(Math.random()-.5)*.1).toFixed(3)),borderColor:lineCol,borderWidth:2,pointRadius:0,tension:.4,fill:true,backgroundColor:lineCol+"18"},
      {data:Array(30).fill(5),borderColor:"#ff244466",borderWidth:1,borderDash:[6,4],pointRadius:0,fill:false},
    ]},options:{...chartDefaults,scales:{...chartDefaults.scales,y:{...chartDefaults.scales.y,min:0,max:6}}}});
  },[state,d.cu]);

  const buildAnalytics=useCallback(()=>{
    if(tab!=="analytics")return;
    buildChart(analyticsRef,analyticsChart,{type:"line",data:{labels:Array.from({length:48},(_,i)=>i%12===0?`Day${Math.floor(i/24)+1} ${i%24===0?"AM":"PM"}`:""),datasets:[
      {data:Array.from({length:48},(_,i)=>+(4.3+Math.sin(i*.3)*.5+(i>36?.06*i:0)+(Math.random()-.5)*.3).toFixed(2)),borderColor:"#00e676",borderWidth:2,pointRadius:0,tension:.3,fill:false},
      {data:Array.from({length:48},(_,i)=>+(39.5+Math.sin(i*.2)*.8-(i>36?.04*i:0)+(Math.random()-.5)*.4).toFixed(2)),borderColor:"#00b4d8",borderWidth:2,pointRadius:0,tension:.3,fill:false,yAxisID:"y2"},
      {data:Array(48).fill(4.5),borderColor:"#00e67644",borderWidth:1,borderDash:[6,4],pointRadius:0,fill:false},
      {data:Array(48).fill(39.8),borderColor:"#00b4d844",borderWidth:1,borderDash:[6,4],pointRadius:0,fill:false,yAxisID:"y2"},
    ]},options:{...chartDefaults,scales:{...chartDefaults.scales,y2:{display:false}}}});
  },[tab,state]);

  useEffect(()=>{
    const load=()=>{buildTrend();buildCusum();buildAnalytics();};
    if(window.Chart)load();
    else{const s=document.createElement("script");s.src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js";s.onload=load;document.head.appendChild(s);}
    return()=>{[trendChart,cusumChart,analyticsChart].forEach(c=>{if(c.current){c.current.destroy();c.current=null;}});};
  },[buildTrend,buildCusum,buildAnalytics,tab]);

  const VITALS=[
    {label:"Motor Vibration (RMS)",val:d.vibe.toFixed(1),unit:"mm/s",icon:"〰️",ok:d.vibe<6},
    {label:"Discharge Pressure",val:d.pres.toFixed(1),unit:"bar",icon:"🔵",ok:d.pres>36},
    {label:"Winding Temperature",val:d.temp,unit:"°C",icon:"🌡️",ok:d.temp<75},
    {label:"Shaft Speed",val:d.rpm,unit:"RPM",icon:"🌀",ok:d.rpm>2950},
  ];
  const showPopup=(state==="WARN"||state==="DANGER")&&!popupDismissed;
  const connectedCount=Object.values(sensorConnected).filter(Boolean).length;
  const SHAP=[{name:"Discharge Pressure drift",val:0.312,c:"#ff8800"},{name:"Suction Pressure oscillation",val:0.287,c:"#ff8800"},{name:"Motor Vibration slope",val:0.198,c:"#ffcc00"},{name:"score_C (chain signal)",val:0.124,c:"#00b4d8"},{name:"Bearing Housing Temp",val:0.079,c:"#7c6fcd"}];

  const sideItems=[
    {id:"dashboard",icon:"🏠",label:"Dashboard"},
    {id:"sensor",icon:"🔌",label:"Sensor Plugin"},
    {id:"analytics",icon:"📈",label:"Analytics"},
    {id:"predictions",icon:"🧠",label:"Predictions"},
    {id:"history",icon:"🕒",label:"History"},
    {id:"settings",icon:"⚙️",label:"Settings"},
    {id:"guide",icon:"📖",label:"Guide & Disclaimer",accent:true},
  ];

  const activeTab = tab==="guide_gate"?"guide":tab;

  return(
    <div style={{display:"flex",background:"#04101e",minHeight:"900px",fontFamily:"'Inter',system-ui,sans-serif",color:"#c0d8f0",fontSize:13,position:"relative",overflow:"hidden"}}>
      <div style={{position:"absolute",inset:0,background:"radial-gradient(ellipse 90% 55% at 50% -5%, rgba(0,90,200,.2) 0%,transparent 65%)",pointerEvents:"none",zIndex:0}}/>

      {showOnboardingPopup&&(
        <OnboardingPopup
          onAccept={()=>{setOnboardingDone(true);setShowOnboardingPopup(false);setTab("dashboard");}}
          onViewGuide={()=>{setShowOnboardingPopup(false);setTab("guide");}}
        />
      )}

      {/* Sidebar */}
      <div style={{width:196,flexShrink:0,background:"rgba(4,12,24,.95)",borderRight:"1px solid rgba(20,55,95,.5)",display:"flex",flexDirection:"column",position:"relative",zIndex:2}}>
        <div style={{padding:"18px 16px 14px",borderBottom:"1px solid rgba(20,55,95,.5)"}}>
          <div style={{fontSize:20,fontWeight:700,color:"#fff",letterSpacing:1}}>⚡ Pump<span style={{color:"#00d4ff"}}>Smart</span></div>
          <div style={{fontSize:10,color:"#1a5060",marginTop:2,letterSpacing:1}}>INDUSTRIAL MONITOR v14.2</div>
        </div>
        {/* v5.0-D: Single-pump badge added */}
        <div style={{margin:"10px 10px",background:"rgba(0,28,56,.55)",border:"1px solid rgba(20,65,105,.45)",borderRadius:8,padding:"10px 12px"}}>
          <div style={{fontSize:11,color:"#00d4ff",fontWeight:600}}>🏭 PUMP-0032</div>
          <div style={{fontSize:9,color:"#1a5060",marginTop:1,fontStyle:"italic"}}>(single-pump v14.2)</div>
          <div style={{fontSize:10,color:"#1a5060",marginTop:2}}>110 kW · 7-stage · 40 bar</div>
          <div style={{display:"flex",alignItems:"center",gap:6,marginTop:7}}>
            <div style={{width:7,height:7,borderRadius:"50%",background:"#00e676",boxShadow:"0 0 7px #00e676"}}/>
            <span style={{fontSize:10,color:"#00e676"}}>Live · {connectedCount}/8 sensors</span>
          </div>
        </div>
        <nav style={{flex:1,padding:"6px 0"}}>
          {sideItems.map(item=>{
            const isGated = !onboardingDone && GATED_TABS.includes(item.id);
            const isActive = activeTab===item.id;
            return(
              <div key={item.id} onClick={()=>handleTabClick(item.id)}
                style={{display:"flex",alignItems:"center",gap:10,padding:"11px 16px",cursor:"pointer",
                  borderLeft:`3px solid ${isActive?"#00d4ff":item.accent?"rgba(0,180,220,.15)":"transparent"}`,
                  background:isActive?"rgba(0,180,220,.07)":item.accent?"rgba(0,160,200,.03)":"transparent",
                  marginTop:item.accent?4:0,borderTop:item.accent?"1px solid rgba(20,55,95,.35)":"none",
                  transition:"all .2s",opacity:isGated?0.45:1}}>
                <span style={{fontSize:15,color:isActive?"#00d4ff":item.accent?"#2a6a7a":"#2a5a70"}}>{item.icon}</span>
                <span style={{fontSize:12,color:isActive?"#00d4ff":item.accent?"#2a7a8a":"#3a6a7a",fontWeight:isActive?500:400,flex:1}}>{item.label}</span>
                {isGated&&<span style={{fontSize:10}}>🔒</span>}
                {item.id==="sensor"&&!isGated&&connectedCount<8&&<span style={{width:6,height:6,borderRadius:"50%",background:"#ff8800",boxShadow:"0 0 5px #ff8800"}}/>}
              </div>
            );
          })}
        </nav>
        {/* Persistent footer disclaimer */}
        <div style={{padding:"12px 14px",borderTop:"1px solid rgba(20,55,95,.4)"}}>
          <div style={{fontSize:10,color:"#1a4050",lineHeight:1.7}}>⚠️ Advisory only<br/>Verify predictions physically<br/>Not for autonomous control</div>
        </div>
      </div>

      {/* Main */}
      <div style={{flex:1,display:"flex",flexDirection:"column",zIndex:1,minWidth:0,position:"relative"}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 18px",background:"rgba(4,12,24,.9)",borderBottom:"1px solid rgba(20,55,95,.5)"}}>
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <span style={{fontSize:15,fontWeight:600,color:"#e0f0ff"}}>
              {activeTab==="sensor"?"🔌 Sensor Plugin":activeTab==="guide"||activeTab==="guide_gate"?"📖 Guide & Disclaimer":activeTab==="analytics"?"📈 Analytics":activeTab==="predictions"?"🧠 Predictions":activeTab==="history"?"🕒 History":activeTab==="settings"?"⚙️ Settings":"🏠 Dashboard"}
            </span>
            {onboardingDone&&(
              <div style={{display:"flex",alignItems:"center",gap:6,padding:"4px 12px",borderRadius:20,background:`${d.color}14`,border:`1px solid ${d.color}40`,transition:"all .4s"}}>
                <div style={{width:6,height:6,borderRadius:"50%",background:d.color,boxShadow:`0 0 7px ${d.color}`}}/>
                <span style={{fontSize:11,fontWeight:600,color:d.color,letterSpacing:.5}}>{d.label}</span>
              </div>
            )}
          </div>
          <div style={{display:"flex",gap:8}}>
            {onboardingDone&&<button onClick={()=>{setOnboardingDone(false);setTab("guide");}} style={{padding:"5px 12px",background:"rgba(20,55,95,.3)",border:"1px solid rgba(40,90,130,.4)",color:"#3a6a7a",borderRadius:8,cursor:"pointer",fontSize:11}}>🔒 Re-acknowledge</button>}
            {onboardingDone&&<button onClick={()=>{const i=STATES.indexOf(state);setState(STATES[(i+1)%STATES.length]);}} style={{padding:"5px 14px",background:"rgba(0,180,220,.1)",border:"1px solid rgba(0,180,220,.3)",color:"#00d4ff",borderRadius:8,cursor:"pointer",fontSize:11,fontWeight:500}}>Demo: cycle state →</button>}
          </div>
        </div>

        <div style={{flex:1,padding:14,overflow:"auto",position:"relative"}}>

          {tab==="guide_gate"&&(
            <div style={{position:"absolute",inset:0,background:"rgba(2,6,14,.85)",backdropFilter:"blur(12px)",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",zIndex:80,animation:"fadeIn .3s ease"}}>
              <div style={{textAlign:"center",maxWidth:420,padding:32}}>
                <div style={{fontSize:52,marginBottom:16}}>🔒</div>
                <div style={{fontSize:18,fontWeight:700,color:"#e0f0ff",marginBottom:10}}>Acknowledgement required</div>
                <div style={{fontSize:13,color:"#5a8aaa",lineHeight:1.7,marginBottom:24}}>You must read the Guide & Disclaimer and tick the acknowledgement box before accessing the monitoring system.</div>
                <button onClick={()=>setTab("guide")} style={{padding:"12px 28px",borderRadius:10,background:"rgba(0,180,220,.18)",border:"1px solid rgba(0,200,220,.4)",color:"#00d4ff",cursor:"pointer",fontSize:14,fontWeight:700}}>
                  📖 Go to Guide & Disclaimer →
                </button>
              </div>
            </div>
          )}

          {(activeTab==="guide"||activeTab==="guide_gate")&&tab!=="guide_gate"&&(
            <GuideTab
              onNavigate={t=>{if(onboardingDone)setTab(t);}}
              onboardingDone={onboardingDone}
              onAcceptFromGuide={()=>{setOnboardingDone(true);setTab("dashboard");}}
            />
          )}

          {/* DASHBOARD */}
          {activeTab==="dashboard"&&onboardingDone&&(
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 255px",gridTemplateRows:"auto auto auto",gap:12}}>
              <Card style={{gridColumn:"1/3"}} animate>
                <CardHdr title="💓 Pump Vitals Hub" right={<span style={{display:"flex",alignItems:"center",gap:5,fontSize:11,color:"#00e676"}}><span style={{width:6,height:6,borderRadius:"50%",background:"#00e676",display:"inline-block",boxShadow:"0 0 6px #00e676"}}/>🟢 Live</span>}/>
                <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10}}>
                  {VITALS.map((v,i)=>(
                    <div key={i} style={{background:"rgba(4,12,26,.75)",border:"1px solid rgba(20,60,100,.4)",borderRadius:10,padding:"12px 14px",animation:`fadeSlide .4s ${i*.08}s both`}}>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
                        <span style={{fontSize:20}}>{v.icon}</span>
                        <span style={{fontSize:10,padding:"2px 7px",borderRadius:10,background:v.ok?"rgba(0,220,100,.08)":"rgba(255,136,0,.1)",color:v.ok?"#00cc77":"#ff8800",border:`1px solid ${v.ok?"rgba(0,220,100,.2)":"rgba(255,136,0,.25)"}`}}>{v.ok?"🟢 Normal":"🟠 Alert"}</span>
                      </div>
                      <div style={{fontSize:10,color:"#2a5a70",marginBottom:5,height:28,lineHeight:1.3}}>{v.label}</div>
                      <div style={{display:"flex",alignItems:"baseline",gap:4}}>
                        <span style={{fontSize:27,fontWeight:700,color:v.ok?"#e0f0ff":d.color,lineHeight:1,transition:"color .4s"}}>{v.val}</span>
                        <span style={{fontSize:11,color:"#3a6a7a"}}>{v.unit}</span>
                      </div>
                      <Sparkline color={v.ok?"#00b4d8":d.color} anomaly={!v.ok}/>
                    </div>
                  ))}
                </div>
              </Card>
              <Card style={{gridColumn:3,gridRow:"1/3",display:"flex",flexDirection:"column",alignItems:"center"}} animate>
                <CardHdr title="🎯 AI Risk Hub"/>
                <RiskGauge state={state}/>
                <div style={{width:"100%",display:"grid",gridTemplateColumns:"1fr 1fr",gap:7,marginTop:12}}>
                  {[{k:"📊 Score A",v:d.sA,c:state==="NORMAL"?"#00e676":d.color},{k:"📈 CUSUM S_n",v:d.cu,c:parseFloat(d.cu)>4?"#ff2244":parseFloat(d.cu)>2?"#ffcc00":"#3a7a9a"},{k:"🎯 Confidence",v:d.cf+"%",c:parseFloat(d.cf)>85?"#00e676":parseFloat(d.cf)>70?"#ff8800":"#ff2244"},{k:"🏷️ Fault class",v:d.fault,c:"#c0d8f0",small:true}].map((it,i)=>(
                    <div key={i} style={{background:"rgba(4,12,26,.75)",border:"1px solid rgba(20,60,100,.35)",borderRadius:8,padding:"8px 10px"}}>
                      <div style={{fontSize:10,color:"#2a5060",marginBottom:3}}>{it.k}</div>
                      <div style={{fontSize:it.small?11:15,fontWeight:600,color:it.c,lineHeight:1.2,transition:"color .4s"}}>{it.v}</div>
                    </div>
                  ))}
                </div>
                <div style={{marginTop:10,padding:"8px 10px",background:"rgba(4,12,26,.5)",border:"1px solid rgba(20,60,100,.3)",borderRadius:8,width:"100%"}}>
                  <div style={{fontSize:10,color:"#2a5060",marginBottom:4}}>📉 score_A vs θ_t</div>
                  <div style={{height:3,background:"rgba(20,60,100,.5)",borderRadius:2}}>
                    <div style={{width:`${Math.min(100,(parseFloat(d.sA)/0.3)*100)}%`,height:3,background:d.color,borderRadius:2,transition:"width .7s ease",boxShadow:`0 0 5px ${d.color}88`}}/>
                  </div>
                </div>
              </Card>
              <Card style={{gridColumn:"1/3",gridRow:2}} animate>
                <CardHdr title="📡 Live Sensor Trends — raw engineering units" right={<div style={{display:"flex",gap:4}}>{["1H","6H","24H","7D"].map((t,i)=><button key={t} style={{padding:"3px 10px",borderRadius:6,fontSize:11,cursor:"pointer",border:"1px solid rgba(20,70,110,.5)",background:i===0?"rgba(0,180,220,.15)":"transparent",color:i===0?"#00d4ff":"#3a6a7a"}}>{t}</button>)}</div>}/>
                <div style={{height:120,position:"relative"}}><canvas ref={trendRef} style={{position:"absolute",inset:0}}/></div>
                <div style={{display:"flex",gap:14,marginTop:10,flexWrap:"wrap"}}>
                  {[{c:"#00b4d8",l:"💧 Discharge pressure (bar)"},{c:"#ff8800",l:"🌡️ Winding temp (°C)"},{c:"#00e676",l:"〰️ Motor vibration (mm/s)"},{c:"rgba(100,160,200,.35)",l:"Steady-state baseline",dash:true}].map((it,i)=>(
                    <div key={i} style={{display:"flex",alignItems:"center",gap:5,fontSize:11,color:"#3a6a7a"}}><div style={{width:14,height:2,background:it.c,borderTop:it.dash?`1px dashed ${it.c}`:"none"}}/>{it.l}</div>
                  ))}
                </div>
              </Card>
              <Card style={{gridColumn:"1/2",gridRow:3}} animate>
                <CardHdr title="🔍 Deep Diagnostics — fault probability"/>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
                  {d.diag.map((item,i)=>(
                    <div key={i} style={{background:"rgba(4,12,26,.75)",border:"1px solid rgba(20,60,100,.4)",borderRadius:10,padding:"10px 12px",display:"flex",gap:10,alignItems:"center",animation:`fadeSlide .5s ${i*.1}s both`}}>
                      <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:2}}>
                        <span style={{fontSize:18}}>{item.icon}</span>
                        <ArcGauge pct={item.p} color={item.c} size={48}/>
                      </div>
                      <div style={{flex:1,minWidth:0}}>
                        <div style={{fontSize:10,color:"#2a5a70",textTransform:"uppercase",letterSpacing:.3,marginBottom:3}}>{item.n}</div>
                        <div style={{fontSize:24,fontWeight:700,color:item.c,lineHeight:1,transition:"color .4s"}}>{item.p}%</div>
                        <div style={{fontSize:10,color:"#2a5060",marginTop:3}}>{item.p>70?"🔴 High — inspect now":item.p>35?"🟠 Monitor closely":item.p>15?"🟡 Elevated":"🟢 Normal"}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
              <Card style={{gridColumn:"2/3",gridRow:3}} animate>
                <CardHdr title="📈 CUSUM Accumulator S_n" right={<span style={{fontSize:10,color:"#2a5060",padding:"2px 7px",background:"rgba(4,12,26,.5)",border:"1px solid rgba(20,60,100,.3)",borderRadius:6}}>L3 · Gradual wear</span>}/>
                <div style={{display:"flex",gap:8,marginBottom:10}}>
                  {[{k:"Current S_n",v:d.cu,c:parseFloat(d.cu)>4?"#ff2244":parseFloat(d.cu)>2?"#ffcc00":"#00e676"},{k:"⛔ Alarm H",v:"5.0",c:"#3a6a7a"}].map((it,i)=>(
                    <div key={i} style={{flex:1,background:"rgba(4,12,26,.75)",border:"1px solid rgba(20,60,100,.35)",borderRadius:8,padding:"7px 10px"}}>
                      <div style={{fontSize:10,color:"#2a5060",marginBottom:3}}>{it.k}</div>
                      <div style={{fontSize:20,fontWeight:700,color:it.c,transition:"color .4s"}}>{it.v}</div>
                    </div>
                  ))}
                </div>
                <div style={{height:90,position:"relative"}}><canvas ref={cusumRef} style={{position:"absolute",inset:0}}/></div>
              </Card>
              <Card style={{gridColumn:3,gridRow:"3/5"}} animate>
                <CardHdr title="🚨 Recent Alerts & Events" right={<span style={{fontSize:11,color:"#00b4d8",cursor:"pointer"}} onClick={()=>setTab("history")}>🕒 View all →</span>}/>
                <div style={{display:"flex",flexDirection:"column",gap:6}}>
                  {d.alerts.map((a,i)=>{const tc=TAGCOLS[a.tag]||"#3a6a7a";return(
                    <div key={i} style={{display:"flex",gap:8,padding:"8px 10px",borderRadius:8,background:"rgba(4,12,26,.65)",border:"1px solid rgba(20,60,100,.35)",alignItems:"flex-start",animation:`fadeSlide .4s ${i*.07}s both`}}>
                      <div style={{width:26,height:26,borderRadius:6,display:"flex",alignItems:"center",justifyContent:"center",fontSize:14,flexShrink:0,background:`${a.col}18`,border:`1px solid ${a.col}44`}}>{a.icon}</div>
                      <div style={{flex:1,minWidth:0}}>
                        <div style={{display:"flex",alignItems:"center",gap:5,marginBottom:3,flexWrap:"wrap"}}>
                          <span style={{fontSize:10,color:"#00b4d8",fontWeight:500}}>{a.time}</span>
                          <span style={{fontSize:10,padding:"1px 7px",borderRadius:8,background:`${tc}18`,color:tc,border:`1px solid ${tc}30`}}>{a.tag}</span>
                        </div>
                        <div style={{fontSize:11,color:"#7aabb8",lineHeight:1.45}}>{a.txt}</div>
                      </div>
                    </div>
                  );})}
                </div>
              </Card>
            </div>
          )}

          {/* SENSOR PLUGIN — v5.0 Concern 3: per-cluster ranges */}
          {activeTab==="sensor"&&onboardingDone&&(
            <div style={{animation:"fadeSlide .35s ease"}}>
              <div style={{marginBottom:14,padding:"12px 16px",background:"rgba(0,100,180,.1)",border:"1px solid rgba(0,140,220,.25)",borderRadius:10,fontSize:12,color:"#6ab0cc",lineHeight:1.7}}>
                <strong style={{color:"#00d4ff"}}>🔌 Day 1 Configuration.</strong> Map each physical sensor to its model channel. After setup, the system runs fully automatically — no further interaction required.
              </div>
              {/* Cluster selector — shows per-cluster ranges */}
              <div style={{marginBottom:14,background:"rgba(6,18,36,.88)",border:"1px solid rgba(20,60,100,.5)",borderRadius:12,padding:"12px 16px"}}>
                <div style={{fontSize:11,color:"#5a9ab0",fontWeight:500,letterSpacing:.8,textTransform:"uppercase",marginBottom:10}}>📊 Showing ranges for operating cluster</div>
                <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                  {CLUSTER_LABELS.map((label,i)=>(
                    <button key={i} onClick={()=>setActiveCluster(i)}
                      style={{padding:"6px 14px",borderRadius:8,cursor:"pointer",fontSize:12,fontWeight:activeCluster===i?600:400,
                        border:`1px solid ${activeCluster===i?CLUSTER_COLORS[i]+"88":"rgba(20,55,95,.5)"}`,
                        background:activeCluster===i?CLUSTER_COLORS[i]+"18":"transparent",
                        color:activeCluster===i?CLUSTER_COLORS[i]:"#3a6a7a",transition:"all .2s"}}>
                      {label}{activeCluster===i?" ◀":""}</button>
                  ))}
                </div>
                {activeCluster===0&&<div style={{marginTop:8,fontSize:11,color:"#ffbb00"}}>⚠️ Startup: Motor vibration 2–4× higher than steady-state — shaft resonance during acceleration. This is correct physics, NOT a fault.</div>}
                {activeCluster===3&&<div style={{marginTop:8,fontSize:11,color:"#00b4d8"}}>ℹ️ Cooldown: Pump casing temperature may drop below ambient by up to 1.4°C due to flash evaporation at 40 bar depressurisation — correct physics (C-09), not a sensor fault.</div>}
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                {SENSORS.map((s,i)=>{const conn=sensorConnected[s.id];return(
                  <div key={s.id} style={{background:"rgba(6,18,36,.88)",border:`1px solid ${conn?"rgba(0,200,120,.2)":"rgba(255,100,0,.25)"}`,borderRadius:12,padding:16,animation:`fadeSlide .4s ${i*.05}s both`}}>
                    <div style={{display:"flex",alignItems:"flex-start",justifyContent:"space-between",marginBottom:10}}>
                      <div>
                        <div style={{fontSize:13,fontWeight:500,color:"#e0f0ff",marginBottom:3}}>{s.icon} {s.name}</div>
                        <div style={{fontSize:10,color:"#2a5a70"}}>{s.desc}</div>
                        {s.note&&<div style={{fontSize:10,color:"#ffbb0099",marginTop:3,fontStyle:"italic"}}>{s.note}</div>}
                      </div>
                      <div style={{display:"flex",alignItems:"center",gap:6,flexShrink:0,marginLeft:10}}>
                        <div style={{width:8,height:8,borderRadius:"50%",background:conn?"#00e676":"#ff6600",boxShadow:conn?"0 0 7px #00e676":"0 0 7px #ff6600",transition:"all .3s"}}/>
                        <span style={{fontSize:10,color:conn?"#00e676":"#ff6600"}}>{conn?"🟢 Connected":"🔴 No signal"}</span>
                      </div>
                    </div>
                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8,marginBottom:10}}>
                      {[{k:"Channel ID",v:s.ch},{k:"Unit",v:s.unit},{k:`Range — ${CLUSTER_LABELS[activeCluster]}`,v:s.clusterRanges[activeCluster],highlight:true}].map((it,j)=>(
                        <div key={j} style={{background:it.highlight?"rgba(0,180,220,.07)":"rgba(4,12,26,.7)",borderRadius:7,padding:"7px 10px",border:it.highlight?`1px solid ${CLUSTER_COLORS[activeCluster]}44`:"1px solid rgba(20,55,95,.4)"}}>
                          <div style={{fontSize:10,color:it.highlight?CLUSTER_COLORS[activeCluster]:"#2a5060",marginBottom:2}}>{it.k}</div>
                          <div style={{fontSize:12,color:it.highlight?CLUSTER_COLORS[activeCluster]:"#c0d8f0",fontFamily:"monospace",fontWeight:it.highlight?600:400}}>{it.v}</div>
                        </div>
                      ))}
                    </div>
                    <div style={{marginBottom:9}}>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}>
                        <span style={{fontSize:10,color:"#2a5060"}}>ISA-37 headroom</span>
                        <span style={{fontSize:10,fontFamily:"monospace",color:s.headroom<HEADROOM_FLAG?"#ff8800":s.headroom<HEADROOM_WARN?"#ffcc00":"#00e676"}}>{(s.headroom*100).toFixed(1)}% {s.headroom<HEADROOM_FLAG?"Near ceiling":s.headroom<HEADROOM_WARN?"Moderate":"Good"}</span>
                      </div>
                      <div style={{height:4,background:"rgba(20,55,95,.4)",borderRadius:2,overflow:"hidden"}}>
                        <div style={{width:`${Math.min(100,(s.headroom/0.25)*100)}%`,height:4,background:s.headroom<HEADROOM_FLAG?"#ff8800":s.headroom<HEADROOM_WARN?"#ffcc00":"#00e676",borderRadius:2}}/>
                      </div>
                    </div>
                    <button onClick={()=>setSensorConnected(p=>({...p,[s.id]:!conn}))} style={{width:"100%",padding:"7px 0",borderRadius:7,cursor:"pointer",fontSize:11,fontWeight:600,border:`1px solid ${conn?"rgba(255,100,0,.4)":"rgba(0,200,120,.35)"}`,background:conn?"rgba(255,100,0,.08)":"rgba(0,200,120,.1)",color:conn?"#ff8800":"#00e676",transition:"all .2s"}}>
                      {conn?"🔴 Disconnect":"🟢 Connect"}
                    </button>
                  </div>
                );})}
              </div>
              {connectedCount<8&&<div style={{marginTop:14,padding:"12px 16px",background:"rgba(255,100,0,.08)",border:"1px solid rgba(255,100,0,.25)",borderRadius:10,fontSize:12,color:"#ff8800"}}>⚠️ {8-connectedCount} sensor(s) disconnected. Model prediction partially paused. Reconnect sensor or confirm pump shutdown.</div>}
            </div>
          )}

          {/* ANALYTICS */}
          {activeTab==="analytics"&&onboardingDone&&(
            <div style={{animation:"fadeSlide .35s ease"}}>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:10,marginBottom:14}}>
                {[{k:"〰️ Avg vibration",v:"4.8 mm/s",c:"#00e676"},{k:"🔵 Avg pressure",v:"38.2 bar",c:"#00b4d8"},{k:"🌡️ Peak temperature",v:"79°C",c:"#ff8800"},{k:"⏱️ Operating hrs",v:"1,840 hrs",c:"#9b7fe8"}].map((it,i)=>(
                  <div key={i} style={{background:"rgba(6,18,36,.88)",border:"1px solid rgba(20,60,100,.4)",borderRadius:10,padding:"12px 14px",animation:`fadeSlide .4s ${i*.07}s both`}}>
                    <div style={{fontSize:10,color:"#2a5060",marginBottom:5,textTransform:"uppercase",letterSpacing:.5}}>{it.k}</div>
                    <div style={{fontSize:22,fontWeight:700,color:it.c}}>{it.v}</div>
                  </div>
                ))}
              </div>
              <Card animate style={{marginBottom:14}}>
                <CardHdr title="📡 48-hour sensor trend — actual vs steady-state baseline"/>
                <div style={{height:160,position:"relative"}}><canvas ref={analyticsRef} style={{position:"absolute",inset:0}}/></div>
              </Card>
              <Card animate>
                <CardHdr title="🔬 SHAP feature importance — top contributors to current prediction"/>
                <div style={{display:"flex",flexDirection:"column",gap:8}}>
                  {SHAP.map((f,i)=>(
                    <div key={i} style={{display:"flex",alignItems:"center",gap:12,animation:`fadeSlide .4s ${i*.08}s both`}}>
                      <div style={{width:200,fontSize:12,color:"#8ab0c8",flexShrink:0}}>{f.name}</div>
                      <div style={{flex:1,height:6,background:"rgba(20,55,95,.4)",borderRadius:3}}>
                        <div style={{width:`${(f.val/.35)*100}%`,height:6,background:f.c,borderRadius:3,boxShadow:`0 0 5px ${f.c}88`,transition:"width .8s ease"}}/>
                      </div>
                      <div style={{width:40,fontSize:12,color:f.c,textAlign:"right",fontFamily:"monospace"}}>{f.val.toFixed(3)}</div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          )}

          {/* PREDICTIONS — v5.0-A: ONLY active-learning write point */}
          {activeTab==="predictions"&&onboardingDone&&(
            <div style={{animation:"fadeSlide .35s ease"}}>
              <Card animate>
                <CardHdr title="🧠 Inference Output — 7-field mandatory output"/>
                <div style={{display:"flex",flexDirection:"column",gap:0}}>
                  {[
                    {n:"01",k:"🏷️ Primary Classification",v:d.fault,c:"#00d4ff"},
                    {n:"02",k:"📊 Probability Matrix",v:`${d.cf}%${parseFloat(d.cf)<70?" ⚠️ LOW CONFIDENCE":""}`,c:parseFloat(d.cf)>85?"#00e676":parseFloat(d.cf)>70?"#ff8800":"#ff2244"},
                    {n:"03",k:"🔬 Physical Interpretation",v:state==="DANGER"?"Seal failure propagating to high-head cavitation — compound fault chain at 40 bar / 450 m head":state==="WARN"?"Suction-side vapour bubble collapse — NPSHa likely below NPSHr at current flow rate":state==="WATCH"?"Sub-threshold bearing drift detected — CUSUM accumulating":"All sensor channels within cluster baseline — normal operation"},
                    {n:"04",k:"📡 Expected Signature",v:state==="NORMAL"?"Flat steady-state across all 8 channels — minor noise only":"Discharge pressure oscillating ±2–4 bar. Suction pressure dropping. Motor vibration rising."},
                    {n:"05",k:"⏳ Consequence Horizon",v:state==="DANGER"?"🔴 URGENT — catastrophic impeller damage within hours at 40 bar":state==="WARN"?"⚠️ Hours to days — impeller erosion accelerates rapidly":"✅ None — continue routine monitoring",c:["WARN","DANGER"].includes(state)?"#ff8800":"#00e676"},
                    {n:"06",k:"🔧 Action Protocol",v:state==="DANGER"?"Reduce pump speed or increase suction head immediately. Replace seal. Do not restart without physical inspection.":state==="WARN"?"Check suction valve. Reduce flow rate. Verify NPSHa ≥ NPSHr + 0.5 m.":"No action required. Maintain routine inspection schedule."},
                    {n:"07",k:"⚠️ Model Disclaimer",v:"Trained on CIRA-anchored physics-synthetic data for 110 kW 7-stage pump at 2980 RPM, 40 bar. Advisory only. Verify physically. Expected F1 on real faults: 0.65–0.85 (C-26).",c:"#3a6a7a"},
                  ].map((row,i)=>(
                    <div key={i} style={{display:"grid",gridTemplateColumns:"40px 200px 1fr",gap:16,padding:"14px 0",borderBottom:"1px solid rgba(20,55,95,.3)",animation:`fadeSlide .4s ${i*.06}s both`}}>
                      <div style={{fontSize:12,color:"#2a5060",fontFamily:"monospace",paddingTop:1}}>{row.n}</div>
                      <div style={{fontSize:11,color:"#3a6a7a",fontWeight:500,paddingTop:2,lineHeight:1.4}}>{row.k}</div>
                      <div style={{fontSize:13,color:row.c||"#c0d8f0",lineHeight:1.6}}>{row.v}</div>
                    </div>
                  ))}
                </div>
                {/* M8p6 sensor health addendum — Field 6 sidecar */}
                <div>{(state==="WARN"||state==="DANGER")&&(<div style={{marginTop:10,padding:"10px 14px",background:"rgba(230,126,34,.07)",border:"1px solid rgba(230,126,34,.3)",borderLeft:"3px solid #e67e22",borderRadius:9}}>
                  <div style={{display:"flex",alignItems:"center",gap:7,marginBottom:5}}>
                    <span style={{fontSize:13}}>{"⚠️"}</span>
                    <span style={{fontSize:11,fontWeight:600,color:"#e67e22"}}>M8p6 Sensor Health Addendum — Field 6 annotation</span>
                    <span style={{fontSize:10,color:"#a06030",marginLeft:"auto",opacity:.7}}>ISA-37 ceiling-approach | Principle 14</span>
                  </div>
                  <div style={{fontSize:12,color:"#d4874a",lineHeight:1.6,marginBottom:5}}>
                    Sensor health: <strong style={{color:"#e67e22"}}>{state==="DANGER"?"Discharge Pressure + Suction Pressure near cluster ceiling":"Discharge Pressure in high_load at 0.94× ceiling"}</strong> — verify transducer calibration before trusting <strong style={{color:"#e67e22"}}>{d.fault}</strong> prediction.
                  </div>
                  <div style={{fontSize:11,color:"#7a5030",lineHeight:1.5}}>Annotation only. Does not change fault label or confidence score. override_existing_prediction: false (locked).</div>
                </div>)}</div>

                {/* v5.0-A: Operator verdict buttons — ONLY active-learning write point */}
                <div style={{marginTop:14,padding:"12px 14px",background:"rgba(4,12,26,.5)",border:"1px solid rgba(20,55,95,.4)",borderRadius:10}}>
                  <div style={{fontSize:11,color:"#3a6a7a",marginBottom:8,lineHeight:1.5}}>
                    📋 <strong style={{color:"#8ab0c8"}}>Rate this prediction</strong> — after physically investigating the fault, submit your professional verdict here. This is recorded in the active-learning repository to improve future model accuracy.
                  </div>
                  {verdictSent?(
                    <div style={{padding:"10px 14px",background:"rgba(0,200,100,.07)",border:"1px solid rgba(0,200,100,.25)",borderRadius:9,fontSize:13,color:"#00e676",display:"flex",gap:8,alignItems:"center"}}>
                      ✅ Verdict recorded: <strong>{verdictSent}</strong> — thank you. Your feedback contributes to model improvement.
                    </div>
                  ):(
                    <div style={{display:"flex",gap:10}}>
                      {[{l:"✅ Prediction correct",c:"#00e676",bg:"rgba(0,200,100,.1)",val:"CORRECT"},{l:"❌ Prediction incorrect",c:"#ff4466",bg:"rgba(200,30,60,.08)",val:"INCORRECT"},{l:"❓ Unsure — inspect further",c:"#ffcc00",bg:"rgba(200,180,0,.08)",val:"UNSURE"}].map((btn,i)=>(
                        <button key={i} onClick={()=>setVerdictSent(btn.val)} style={{flex:1,padding:"10px 6px",borderRadius:8,cursor:"pointer",fontSize:12,fontWeight:500,border:`1px solid ${btn.c}44`,background:btn.bg,color:btn.c}}>{btn.l}</button>
                      ))}
                    </div>
                  )}
                </div>
              </Card>
            </div>
          )}

          {/* HISTORY */}
          {activeTab==="history"&&onboardingDone&&(
            <div style={{animation:"fadeSlide .35s ease"}}>
              <Card animate>
                <CardHdr title="🕒 Full Event History — audit trail"/>
                <div style={{display:"grid",gridTemplateColumns:"110px 100px 100px 1fr 220px 110px",gap:16,padding:"0 0 12px",borderBottom:"1px solid rgba(20,55,95,.4)",fontSize:11,color:"#2a5060",textTransform:"uppercase",letterSpacing:.8}}>
                  <span>📅 Date</span><span>🕒 Time</span><span>🚦 Status</span><span>🏷️ Classification</span><span>🔧 Action taken</span><span>✅ Verified</span>
                </div>
                {HISTORY_EVENTS.map((ev,i)=>{const sc=STATECOLS[ev.state]||"#3a6a7a";return(
                  <div key={i} style={{display:"grid",gridTemplateColumns:"110px 100px 100px 1fr 220px 110px",gap:16,padding:"12px 0",borderBottom:"1px solid rgba(20,55,95,.25)",alignItems:"center",animation:`fadeSlide .3s ${i*.05}s both`,fontSize:13}}>
                    <div style={{color:"#5a9ab0",fontFamily:"monospace",fontSize:11}}>{ev.date}</div>
                    <div style={{color:"#00b4d8",fontFamily:"monospace",fontSize:11}}>{ev.time}</div>
                    <div><span style={{fontSize:10,padding:"2px 8px",borderRadius:6,background:`${sc}18`,color:sc,border:`1px solid ${sc}30`,fontWeight:600}}>{ev.state}</span></div>
                    <div style={{color:"#c0d8f0"}}>{ev.fault}</div>
                    <div style={{color:"#7aabb8",fontSize:11,lineHeight:1.4}}>{ev.action}</div>
                    <div style={{fontSize:11,fontWeight:600,color:ev.verified.includes("✓")?"#00e676":ev.verified==="Pending"?"#ffcc00":"#3a6a7a"}}>{ev.verified}</div>
                  </div>
                );})}
              </Card>
            </div>
          )}

          {/* SETTINGS */}
          {activeTab==="settings"&&onboardingDone&&(
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,animation:"fadeSlide .35s ease"}}>
              <Card animate>
                <CardHdr title="⚙️ Engine Parameters — locked"/>
                {[{k:"🔒 M4 LSTM-AE threshold",v:"0.110058",note:"L1 — PERMANENTLY LOCKED"},{k:"🧠 TCN-AE architecture",v:"v14.2",note:"L2 compound fault"},{k:"🏷️ XGBoost classes",v:"22 labels",note:"Groups A–E"},{k:"📈 CUSUM alarm H",v:"5.0",note:"Label 21 detection"},{k:"📊 θ_initial",v:"1.881275",note:"L4 adaptive threshold"},{k:"🔬 Training basis",v:"CIRA synthetic",note:"110 kW, 2980 RPM, 40 bar"}].map((it,i)=>(
                  <div key={i} style={{paddingBottom:12,marginBottom:12,borderBottom:"1px solid rgba(20,55,95,.3)"}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                      <div style={{fontSize:13,color:"#8ab0c8"}}>{it.k}</div>
                      <div style={{fontSize:13,color:"#00d4ff",fontFamily:"monospace"}}>{it.v}</div>
                    </div>
                    <div style={{fontSize:10,color:"#2a5060",marginTop:3}}>{it.note}</div>
                  </div>
                ))}
              </Card>
              <Card animate>
                <CardHdr title="🚀 Deployment Operations"/>
                <div style={{marginBottom:14,padding:"10px 12px",background:"rgba(0,100,180,.08)",border:"1px solid rgba(0,140,220,.2)",borderRadius:8,fontSize:12,color:"#6ab0cc",lineHeight:1.7}}>🟢 System active. {connectedCount}/8 sensors online. Commissioning mode OFF.</div>
                {[{k:"📅 Monitoring started",v:"2026-05-10 08:00"},{k:"🔌 Active sensors",v:`${connectedCount} / 8`},{k:"📐 Normalization",v:"M3 v14.2 — locked"},{k:"🚀 Commissioning",v:"Disabled",c:"#00e676"},{k:"💾 Data persistence",v:"HF Datasets API",c:"#00b4d8"}].map((it,i)=>(
                  <div key={i} style={{display:"flex",justifyContent:"space-between",padding:"9px 0",borderBottom:"1px solid rgba(20,55,95,.3)"}}>
                    <div style={{fontSize:13,color:"#5a9ab0"}}>{it.k}</div><div style={{fontSize:13,color:it.c||"#c0d8f0",fontFamily:"monospace"}}>{it.v}</div>
                  </div>
                ))}
                <button style={{width:"100%",marginTop:14,padding:"9px",borderRadius:8,cursor:"pointer",fontSize:12,fontWeight:500,border:"1px solid rgba(255,136,0,.3)",background:"rgba(255,136,0,.07)",color:"#ff8800"}}>🔄 Enable commissioning mode</button>
              </Card>
              <Card animate style={{gridColumn:"1/3"}}>
                <CardHdr title="📄 System Scope & Disclaimer"/>
                <div style={{fontSize:12,color:"#4a7a8a",lineHeight:1.9}}>
                  ⚙️ PumpSmart v14.2 — single-pump monitoring system. Trained on CIRA-anchored physics-synthetic data for a <strong style={{color:"#c0d8f0"}}>110 kW, 7-stage centrifugal pump at 2980 RPM, 40 bar, 45 m³/h.</strong> All predictions are <strong style={{color:"#c0d8f0"}}>advisory only</strong> — verify physically before any maintenance action. Shadow mode — not for autonomous control. Cross-pump effects not modelled. Expected real-world F1: 0.65–0.85 (C-26). Confidence ≥85% recommended before acting.
                  <span style={{cursor:"pointer",color:"#00b4d8",marginLeft:8}} onClick={()=>setTab("guide")}> → 📖 Full Guide & Disclaimer</span>
                </div>
              </Card>
            </div>
          )}

        </div>
      </div>

      {/* Fault popup — v5.0-A: Acknowledge = operational reset ONLY */}
      {onboardingDone&&showPopup&&(
        <div style={{position:"absolute",inset:0,background:"rgba(2,8,18,.72)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:50,animation:"fadeIn .3s ease"}}>
          <div style={{background:"#060f1e",border:`1px solid ${d.color}55`,borderRadius:14,padding:24,maxWidth:480,width:"90%",boxShadow:`0 0 50px ${d.color}1a`,animation:"scaleIn .3s ease"}}>
            <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:16}}>
              <div style={{width:42,height:42,borderRadius:10,background:`${d.color}18`,border:`1px solid ${d.color}44`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:22,flexShrink:0}}>{state==="DANGER"?"🔴":"⚠️"}</div>
              <div>
                <div style={{fontSize:16,fontWeight:700,color:d.color}}>{d.label} — {d.fault}</div>
                <div style={{fontSize:11,color:"#3a6a7a",marginTop:2}}>Confidence {d.cf}% · {new Date().toLocaleTimeString()}</div>
              </div>
            </div>
            {[{k:"🔬 Physical condition",v:state==="DANGER"?"Seal failure driving high-head cavitation — compound chain at 40 bar":"Suction-side vapour bubble collapse — NPSHa likely below NPSHr"},{k:"⏳ Risk if ignored",v:state==="DANGER"?"URGENT — catastrophic impeller damage within hours":"Hours to days — impeller erosion accelerates rapidly at 40 bar"},{k:"🔧 Recommended action",v:state==="DANGER"?"Reduce pump speed or increase suction head immediately. Replace seal.":"Check suction valve. Reduce flow rate. Verify NPSHa ≥ NPSHr + 0.5 m."}].map((row,i)=>(
              <div key={i} style={{display:"grid",gridTemplateColumns:"140px 1fr",gap:10,paddingBottom:8,borderBottom:"1px solid rgba(20,55,95,.3)",marginBottom:8}}>
                <div style={{fontSize:11,color:"#3a6a7a",paddingTop:1,fontWeight:500}}>{row.k}</div>
                <div style={{fontSize:12,color:row.k.includes("Risk")?"#ff8800":"#c0d8f0",lineHeight:1.5}}>{row.v}</div>
              </div>
            ))}
            {/* v5.0-A: clarified popup text — ISA-18.2 alarm management */}
            <div style={{marginBottom:12,padding:"8px 12px",background:"rgba(0,150,210,.06)",border:"1px solid rgba(0,150,210,.2)",borderRadius:8,fontSize:11,color:"#4a8aaa",lineHeight:1.6}}>
              ℹ️ <strong style={{color:"#6ab0cc"}}>Acknowledge</strong> resets the alarm state (CUSUM, z_t buffer, rolling baseline). To rate the model's accuracy, use the <strong style={{color:"#6ab0cc"}}>Predictions tab</strong> once you have physically verified the fault.
            </div>
            <div style={{display:"flex",gap:10,marginTop:4}}>
              <button onClick={()=>setPopupDismissed(true)} style={{flex:2,padding:10,borderRadius:8,background:"rgba(0,100,180,.2)",border:"1px solid rgba(0,140,220,.4)",color:"#00b4d8",cursor:"pointer",fontSize:13,fontWeight:600}}>✅ Acknowledge — reset alarm state</button>
              <button onClick={()=>setPopupDismissed(true)} style={{flex:1,padding:"10px 12px",borderRadius:8,background:"rgba(30,8,14,.5)",border:"1px solid rgba(150,20,40,.3)",color:"#ff4466",cursor:"pointer",fontSize:13}}>❌ Dismiss</button>
            </div>
            <div style={{marginTop:10,fontSize:10,color:"#1a4050",lineHeight:1.6}}>⚠️ Advisory only — verify physically. <span style={{color:"#3a6a7a",cursor:"pointer"}} onClick={()=>{setPopupDismissed(true);setTab("guide");}}>→ 📖 View full disclaimer</span></div>
          </div>
        </div>
      )}

      <style>{`@keyframes fadeSlide{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}@keyframes fadeIn{from{opacity:0}to{opacity:1}}@keyframes scaleIn{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:scale(1)}}`}</style>
    </div>
  );
}
