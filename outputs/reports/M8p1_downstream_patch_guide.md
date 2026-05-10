# Downstream code patch guide — apply manually after review

The canonical registry is now `models/unit_registry_canonical.json`.
The three legacy files have been backed up with `.bak_pre_canonical` suffix.
**Do not delete the legacy files yet** — keep them for one full sprint to confirm
no script silently reads them, then delete after M9.

## Required reads to update

### 1. `src/module_05_physics_engine.py`
Replace the inline UNIT_REGISTRY dict definition (Section 8) with a load:

```python
# OLD:
# UNIT_REGISTRY = { ... inline dict ... }
# with open(unit_reg_path, 'w') as f: json.dump(UNIT_REGISTRY, f, indent=2)

# NEW:
unit_reg_canonical = MODEL_DIR / "unit_registry_canonical.json"
if unit_reg_canonical.exists():
    with open(unit_reg_canonical) as f:
        UNIT_REGISTRY = json.load(f)
else:
    raise FileNotFoundError(
        "Canonical unit registry missing. Run module_08p1 first."
    )
```

### 2. M10 Flask `app/routes/physics.py` (when written)
Read units for display from the canonical registry only:

```python
with open(MODEL_DIR / "unit_registry_canonical.json") as f:
    REGISTRY = json.load(f)
display_unit = REGISTRY['channels'][channel_name]['raw_unit_symbol']
```

### 3. M11 SCADA ingestion adapter (when written)
This is the *critical* deployment-side check. Before any sample enters
the inference pipeline:

```python
def validate_incoming_units(scada_payload, registry):
    for channel, value_dict in scada_payload.items():
        if channel not in registry['channels']:
            continue   # ignore unknown channels
        expected = registry['channels'][channel]['raw_unit_per_CIRA']
        actual   = value_dict.get('unit', None)
        if actual is None:
            raise ValueError(
                f"SCADA payload for {channel} has no unit field. "
                f"Cannot validate against registry. HALT."
            )
        if actual != expected:
            raise ValueError(
                f"Unit mismatch on {channel}: SCADA reports {actual}, "
                f"registry expects {expected}. NEVER auto-convert. HALT."
            )
```

## NOT touched (intentional)
- `M3_normalization_config.json` — values are dimensionless ratios; unit-invariant
- M4 weights, M7 weights, M8 weights — trained on normalised inputs; unit-invariant
- All M5 nameplate equations — already in correct SI units
