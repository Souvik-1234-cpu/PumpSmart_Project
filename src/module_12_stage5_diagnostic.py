# src/module_12_stage5_diagnostic.py
# M12 Stage 5 — pre-build diagnostic. Read-only. Dumps runtime contracts of the
# large artifacts (M7 model/label-map/feature-spec, M4/M8 PKLs) + generator API
# so the validation rig matches the live system exactly. NOTHING is modified.
import os, sys, json, importlib.util, inspect, hashlib, traceback
from pathlib import Path


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def hr(t): print("\n" + "="*70 + f"\n{t}\n" + "="*70)
def safe(fn, *a, **k):
    try: return fn(*a, **k)
    except Exception as e: return f"<ERR {type(e).__name__}: {e}>"

# ── 0. Resolve config ────────────────────────────────────────────────────────
hr("0. CONFIG")
try:
    import config
    for n in dir(config):
        if n.isupper():
            v = getattr(config, n)
            print(f"  config.{n} = {v!r}")
except Exception:
    print("  config import failed:"); traceback.print_exc()
    sys.exit("ABORT: fix config import path first (run from project root).")

def D(name, default=None):
    return getattr(config, name, default)

MODEL_DIR  = D("MODEL_DIR");  OUTPUT_DIR = D("OUTPUT_DIR")
SYNTH_DIR  = D("SYNTH_DIR");  NORM_DIR   = D("NORM_DIR")

# ── 1. Enumerate model dir with sizes + sha256(first 1MB) ────────────────────
hr("1. MODEL DIR INVENTORY")
def sha1mb(p):
    h = hashlib.sha256()
    with open(p, "rb") as f: h.update(f.read(1_000_000))
    return h.hexdigest()[:16]
for base in [MODEL_DIR, SYNTH_DIR, NORM_DIR]:
    if not base: continue
    print(f"\n[{base}]")
    for p in sorted(Path(base).rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            tag = sha1mb(p) if sz < 50_000_000 else "(>50MB skip-hash)"
            print(f"  {sz:>12,}  {tag}  {p.relative_to(base)}")

# ── 2. M7 XGBoost model contract ─────────────────────────────────────────────
hr("2. M7 XGBOOST CONTRACT")
def find(base, *names):
    for n in names:
        for p in Path(base).rglob(n):
            return p
    return None
m7 = find(MODEL_DIR, "*m7*v3*.json", "*m7*v3*.pkl", "*xgb*v3*.json",
          "*xgb*.json", "*m7*.pkl", "*xgb*.pkl")
print("  M7 file:", m7)
if m7:
    try:
        import xgboost as xgb
        print("  xgboost", xgb.__version__)
        if str(m7).endswith(".json"):
            b = xgb.Booster(); b.load_model(str(m7))
            print("  n_features (booster):", safe(lambda: b.num_features()))
            cfg = safe(lambda: json.loads(b.save_config()))
            if isinstance(cfg, dict):
                lp = cfg.get("learner", {}).get("learner_model_param", {})
                print("  num_class:", lp.get("num_class"))
                print("  base_score:", lp.get("base_score"))
                print("  objective:", cfg.get("learner", {}).get("objective"))
            fn = safe(lambda: b.feature_names)
            print("  feature_names:", fn if fn else "<none embedded>")
        else:
            import pickle
            with open(m7, "rb") as f: obj = pickle.load(f)
            print("  pickled type:", type(obj))
            for attr in ("n_classes_", "classes_", "n_features_in_",
                         "feature_names_in_", "get_booster"):
                if hasattr(obj, attr):
                    v = getattr(obj, attr)
                    print(f"  .{attr}:", safe(v) if callable(v) else v)
    except Exception:
        traceback.print_exc()

# ── 3. Label map + calibration + feature spec ────────────────────────────────
hr("3. LABEL MAP / CALIBRATION / FEATURE SPEC")
for pat in ["*label*map*.json", "*labels*.json", "*calib*.json",
            "*temperature*.json", "*feature*spec*.json", "*feature*cols*.json",
            "*m7*meta*.json", "*m7*feature*.json"]:
    for p in Path(MODEL_DIR).rglob(pat):
        print(f"\n[{p.name}]")
        print("  ", safe(lambda: json.dumps(json.load(open(p, encoding="utf-8")),
                                            indent=2)[:1500]))

# ── 4. Generator API (m6b_physics_lib + adversarial generator) ───────────────
hr("4. GENERATOR API")
def dump_module_api(modpath):
    p = Path(modpath)
    if not p.exists():
        print(f"  MISSING: {modpath}"); return
    print(f"\n[{p.name}]")
    src = p.read_text(encoding="utf-8", errors="replace")
    # top-level defs + signatures
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("def ") or s.startswith("class ") or \
           (s.isupper() and "=" in s and not s.startswith("#")):
            print("   ", s[:140])
for cand in ["m6b_physics_lib.py", "src/m6b_physics_lib.py",
             "app/runtime/m6b_physics_lib.py",
             "module_12a_adversarial_generator.py",
             "src/module_12a_adversarial_generator.py",
             "module_12b_adversarial_runner.py",
             "src/module_12b_adversarial_runner.py"]:
    dump_module_api(cand)

# ── 5. Adversarial runner: extract its conventions verbatim ──────────────────
hr("5. RUNNER CONVENTIONS (label map, warmup, detection signal, endpoints)")
rp = None
for c in ["src/module_12b_adversarial_runner.py", "module_12b_adversarial_runner.py"]:
    if Path(c).exists(): rp = Path(c); break
if rp:
    src = rp.read_text(encoding="utf-8", errors="replace")
    for kw in ["WARMUP", "warmup", "sigma", "0.045", "raw_alert_state",
               "score_A_max", "0.05", "test_reset_latch", "LABEL", "label_map",
               "/api/", "base_url", "reset_zt", "GROUP_B", "threshold"]:
        hits = [l for l in src.splitlines() if kw in l]
        if hits:
            print(f"\n  ── '{kw}' ──")
            for l in hits[:8]: print("   ", l.strip()[:140])

# ── 6. fault_state_tracker contract ──────────────────────────────────────────
hr("6. FAULT STATE TRACKER API")
dump_module_api("app/runtime/fault_state_tracker.py")
dump_module_api("fault_state_tracker.py")

# ── 7. A sample synthetic sequence shape (if any pkl present) ────────────────
hr("7. SAMPLE SYNTHETIC SEQUENCE SHAPE")
for pat in ["M6B_combined_sequences.pkl", "*combined*sequences*.pkl",
            "*sequences*.pkl"]:
    p = find(SYNTH_DIR or MODEL_DIR, pat)
    if p:
        try:
            import pickle, numpy as np
            with open(p, "rb") as f: obj = pickle.load(f)
            print(f"  [{p.name}] type={type(obj)}")
            if isinstance(obj, dict):
                for k in list(obj)[:6]:
                    v = obj[k]
                    sh = getattr(v, "shape", None) or safe(len, v)
                    print(f"    key={k!r:>20}  type={type(v).__name__}  shape/len={sh}")
            elif isinstance(obj, (list, tuple)):
                print(f"    len={len(obj)}  first={type(obj[0])}")
                print(f"    first shape:", getattr(obj[0], "shape", None))
        except Exception:
            traceback.print_exc()
        break

# ── 8. M4 / M8 torch artifact shapes ─────────────────────────────────────────
hr("8. M4 / M8 TORCH ARTIFACT KEYS")
for pat in ["*m4*.pt", "*m4*.pth", "*lstm*ae*.pt", "*tcn*.pt", "*m8*.pt"]:
    for p in Path(MODEL_DIR).rglob(pat):
        try:
            import torch
            sd = torch.load(p, map_location="cpu")
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            print(f"\n[{p.name}] {len(sd)} tensors")
            for k, v in list(sd.items())[:12]:
                print(f"    {k:<40} {tuple(v.shape)}")
        except Exception:
            print(f"\n[{p.name}] load err"); traceback.print_exc()

print("\n\n=== DIAGNOSTIC COMPLETE — paste ALL stdout above ===")