# 🚨 Template Mismatch Safety: Detection and Prevention

## Overview

This document describes the **template mismatch risk** in the Sensitivity pipeline and the **warnings and detection mechanisms** added to prevent it.

---

## 🔥 The Problem: Template Mismatch

The Sensitivity pipeline generates templates in **multiple phases**:

1. **Phase 1**: `03_template_compute.py --template background` → Generates **background templates**
2. **Phase 2**: `04_best_cuts.py` → Selects the **best cut** and writes `highest_SENSITIVITY.pkl`
3. **Phase 3**: `03_template_compute.py --template signal` → Generates **signal templates** for the best cut
4. **Phase 4**: `06_significance.py` → Uses **both templates** for final analysis

### ❌ The Mismatch Scenario

If **Phase 1** runs **AFTER** a best-cut file already exists (from a previous run):

1. **Phase 1** sees the existing `highest_SENSITIVITY.pkl` → generates background templates for **Cut A (OLD)**
2. **Phase 2** re-runs → selects **Cut B (NEW, different from A)** → overwrites `highest_SENSITIVITY.pkl`
3. **Phase 3** uses the **NEW** best-cut file → generates signal templates for **Cut B (NEW)**
4. **Phase 4** combines:
   - Background templates: **Cut A (OLD)**
   - Signal templates: **Cut B (NEW)**
   - **❌ RESULT: MISMATCHED TEMPLATES → INVALID RESULTS**

---

## ✅ Solutions Implemented

### 1. **`01_background_template.py` (and `03_template_compute.py`)**

#### Added Arguments:
- `--template {background,signal}` → Distinguishes template type
- `--force-all-cuts` → **RECOMMENDED** for background templates to force all cuts

#### Added Safety Checks:

##### a) **Critical Warning for Background Templates**
```python
# If best-cut file exists and we're generating BACKGROUND templates:
rprint(
    "[red][CRITICAL WARNING][/red] BEST-CUT FILE EXISTS but we are generating "
    "BACKGROUND templates. This can lead to a MISMATCH if 04_best_cuts.py selects "
    "a different cut later in the pipeline."
)
```

##### b) **Pipeline Order Detection**
```python
# Check if signal templates already exist (indicates 04_best_cuts.py already ran)
if signal_template_files:
    rprint(
        "[red][CRITICAL][/red] Signal templates already exist. "
        "This suggests 04_best_cuts.py has already run, but you are now generating "
        "BACKGROUND templates. This is the WRONG ORDER."
    )
```

##### c) **Cut Consistency Validation**
```python
if args.template == "background" and len(cut_entries) == 1:
    rprint(
        "[red][CRITICAL][/red] Generating background templates for ONLY ONE cut. "
        "This is DANGEROUS if 04_best_cuts.py has not yet run."
    )
```

---

### 2. **`run_sensitivity.py` (Main Pipeline Orchestrator)**

#### Added Global Warning:
```python
# At startup, if --computation + --no-rewrite + Sensitivity:
if args.computation and not args.rewrite and "Sensitivity" in args.analysis:
    rprint(
        "[red][CRITICAL WARNING][/red] GLOBAL FLAG CONFLICT DETECTED:"
        "  --computation is enabled (will generate new outputs)"
        "  --no-rewrite is set (will NOT overwrite existing files)"
        "\n"
        "RECOMMENDATION: Use --rewrite OR (--skip-templates AND --skip_best_cuts)"
    )
```

#### Added Pipeline Order Reminder:
```python
if "Sensitivity" in args.analysis and args.computation:
    rprint(
        "[cyan][PIPELINE ORDER REMINDER][/cyan] For Sensitivity analysis:"
        "  PHASE 1: 03_template_compute.py --template background (ALL cuts)"
        "  PHASE 2: 04_best_cuts.py (selects best cut)"
        "  PHASE 3: 03_template_compute.py --template signal (best cut ONLY)"
        "  PHASE 4: 06_significance.py (uses SAME cut for both)"
        "\n"
        "  ⚠️  MISMATCH RISK: If Phase 1 runs AFTER best-cut file exists..."
    )
```

#### Added State Detection Function:
```python
def warn_template_mismatch_risk(config, folder, name):
    """Detects pre-existing files that could cause mismatches."""
    # Checks for:
    # - Best-cut file exists but no background templates (incomplete state)
    # - Both best-cut and background templates exist (potential stale state)
    # - Signal templates exist but no best-cut file (incomplete state)
```

#### Added Phase-Specific Warnings:
```python
# In run_sensitivity_stage():
if args.computation and not args.skip_templates and not args.skip_best_cuts:
    if not args.rewrite:
        rprint("[red][CRITICAL WARNING][/red] --no-rewrite with template generation..."
    
    # Check for existing best-cut files
    best_cut_files = glob.glob(best_cut_pattern)
    if best_cut_files:
        rprint(
            "[red][CRITICAL WARNING][/red] Existing best-cut file(s) found..."
            "Background templates may only cover OLD best cut."
        )
```

---

## 🛡️ How to Use Safely

### ✅ **SAFE: Full Pipeline Run (Recommended)**
```bash
# Always use --rewrite for full consistency
python3 src/pipelines/run_sensitivity.py \
    --analysis Sensitivity \
    --config hd_1x2x6_centralAPA \
    --rewrite  # ← CRITICAL: Ensures all files regenerated together
```

### ✅ **SAFE: Skip All Computation**
```bash
# Only generate plots from existing consistent files
python3 src/pipelines/run_sensitivity.py \
    --analysis Sensitivity \
    --no-computation \
    --no-rewrite
```

### ✅ **SAFE: Reuse Existing Files Consistently**
```bash
# Skip ALL generation steps (templates AND best cuts)
python3 src/pipelines/run_sensitivity.py \
    --analysis Sensitivity \
    --skip-templates \
    --skip_best_cuts \
    --no-rewrite
```

### ❌ **DANGEROUS: Partial Regeneration**
```bash
# ❌ AVOID: Generating new templates without --rewrite
python3 src/pipelines/run_sensitivity.py \
    --analysis Sensitivity \
    --no-rewrite  # ← DANGEROUS with --computation

# ❌ AVOID: Generating templates but skipping best-cut selection
python3 src/pipelines/run_sensitivity.py \
    --analysis Sensitivity \
    --skip_best_cuts  # ← May use stale best-cut file
```

---

## 🔍 Warning Messages Reference

| Warning Level | Message | Meaning | Action |
|---------------|---------|---------|--------|
| `[red][CRITICAL][/red]` | Best-cut file exists but generating background templates | Potential stale state | Use `--rewrite` or `--force-all-cuts` |
| `[red][CRITICAL][/red]` | Signal templates exist but no best-cut file | Incomplete state | Run `04_best_cuts.py` first |
| `[red][CRITICAL][/red]` | Global flag conflict (`--computation` + `--no-rewrite`) | Will generate mismatched files | Use `--rewrite` or skip all generation |
| `[yellow][WARNING][/yellow]` | Unable to load best-cut map | First run or missing file | Normal (will use all cuts) |
| `[yellow][WARNING][/yellow]` | Falling back to N cut triplets | No best-cut file found | Normal for first run |
| `[cyan][INFO][/cyan]` | Using all N cut triplets | Safe: all cuts being generated | Proceed normally |
| `[cyan][PIPELINE ORDER REMINDER][/cyan]` | Pipeline phase order | Educational reminder | Review pipeline logic |

---

## 📋 Files Modified

1. **`src/physics/sensitivity/01_background_template.py`**
   - Added `--template` and `--force-all-cuts` arguments
   - Added `_load_best_cut_map_safe()` function with warnings
   - Added `_validate_cut_consistency()` function
   - Added pipeline order detection
   - Added cut selection warnings

2. **`src/pipelines/run_sensitivity.py`**
   - Added `import glob`
   - Added `warn_template_mismatch_risk()` function
   - Added global flag conflict detection
   - Added pipeline order reminder
   - Added per-phase warnings in `run_sensitivity_stage()`

---

## 🎯 Best Practices

1. **Always use `--rewrite`** when running full computation
2. **Never run `01_background_template.py` separately** — use the pipeline
3. **For background templates: use `--force-all-cuts`** to ensure all cuts are generated
4. **For signal templates: best-cut file is required** — ensure `04_best_cuts.py` runs first
5. **Validate state before running**: Check for existing best-cut and template files

---

## 📞 Getting Help

If you see a `[red][CRITICAL]` warning:
1. **STOP** the pipeline
2. Check which files exist in your output directory
3. Use `--rewrite` to start fresh
4. Or manually clean up stale files before re-running

---

## 🔗 See Also

- `src/physics/sensitivity/04_best_cuts.py` — Best-cut selection logic
- `src/physics/sensitivity/03_template_compute.py` — Template generation (primary script)
- `src/physics/sensitivity/06_significance.py` — Final sensitivity analysis
