# Signal 100x Error Diagnostic Guide

## Quick Start

Run all diagnostics:
```bash
bash run_diagnostics.sh
```

Or focus on regression testing:
```bash
bash compare_weighting_git.sh
python3 test_weighting_regression.py
```

## Individual Tests

### 1. **Rebinning Logic Test** (Most Critical)
```bash
python3 test_rebin_logic.py
```

**What it checks:**
- Input: 2000×120 oscillation matrix (nadir × energy)
- Rebin by factor of 3
- Verify output sum equals input sum
- If ratio is ~100x, rebinning bug confirmed

**Expected result:**
```
Ratio (actual/expected): 1.000000  ✓ PASS
```

**If failed (100x):**
- Bug is in rebin_hist2d axis handling
- Check: Did the lib/dataframe.py fix actually get applied?
- Verify: `grep "len(y) // rebin" lib/dataframe.py`

---

### 2. **Oscillation Weighting Test**
```bash
python3 test_oscillation_weighting.py
```

**What it checks:**
- Compute oscillation probabilities (nufast backend)
- Load nadir PDF (file and nufast versions)
- Apply nadir weighting via combine_day_night()
- Verify normalization

**Expected result:**
```
Nadir PDF sum: 1.000000 (should be 1.0)
Combined sum: ~6000.000000 (120 bins × 40 bins × 0.5 avg P_ee)
Ratio: 1.00x
```

**If ratio is ~100x:**
- Oscillation weighting is the problem
- Could be: nadir PDF not normalized, or applied twice, or wrong source

---

### 3. **Git Comparison (Regression Detection)**
```bash
bash compare_weighting_git.sh
```

**What it shows:**
- OLD approach (bb5d270): process_oscillation_map()
- NEW approach (HEAD): combine_day_night() + compute backends
- Side-by-side code comparison
- Key differences highlighted

**What to look for:**
- Did nadir PDF source change?
- Did normalization step get removed/added?
- Did energy grid change?

**To run git bisect:**
```bash
git bisect start
git bisect bad HEAD
git bisect good bb5d270
# Then run sensitivity analysis on each commit
```

---

### 4. **Signal Template Comparison**
```bash
bash diagnostic_signal_check.sh
```

**Shows:**
- Signal template file timestamps (confirms regeneration)
- Signal template statistics (sum, mean, shape)
- Oscillation PDF normalization check
- Detector mass/exposure settings

**Key values to check:**
- Total signal events in template
- Were templates regenerated? (timestamp check)
- Are nadir PDFs normalized to 1.0?

---

### 5. **Version Comparison** (If old data available)
```bash
# First, update path in script:
# old_path = Path("/path/to/old/SENSITIVITY/truncated/solarenergy")

python3 compare_signal_versions.py
```

**What it does:**
- Loads new signal template
- Loads old signal template
- Compares total signal
- Reports ratio (new/old)

**Expected result:**
```
Ratio (new/old): 1.00x (within 5%)
```

**If ratio is ~100x:**
- Signal is definitely broken
- Either rebinning or weighting is the culprit

---

## Diagnosis Decision Tree

```
Run: python3 test_rebin_logic.py
  │
  ├─ Ratio ~100x?
  │  └─ YES → BUG IN REBINNING
  │     • Check lib/dataframe.py line 121
  │     • Verify axis swap: len(y) // rebin vs len(x) // rebin
  │     • Regenerate templates
  │
  └─ Ratio ~1.0? → Rebinning is fine
     │
     └─ Run: python3 test_oscillation_weighting.py
        │
        ├─ Ratio ~100x?
        │  └─ YES → BUG IN OSCILLATION WEIGHTING
        │     • Check nadir PDF normalization
        │     • Check if combine_day_night applies weights correctly
        │     • Run: bash compare_weighting_git.sh
        │
        └─ Ratio ~1.0? → Weighting is fine
           │
           └─ Run: python3 compare_signal_versions.py
              │
              ├─ New/Old ratio ~100x?
              │  └─ YES → SIGNAL NORMALIZATION CHANGED
              │     • Check template scaling factors
              │     • Check exposure/detector mass
              │
              └─ New/Old ratio ~1.0? → NO REGRESSION DETECTED
                 └─ Check other analysis components
                    • Background loading
                    • Chi² computation
                    • Significance calculation
```

## Common Culprits

### If rebinning is the issue:
```python
# Bad (axes transposed):
z[: len(x) // rebin * rebin, : len(y) // rebin * rebin]

# Good (axes correct):
z[: len(y) // rebin * rebin, : len(x) // rebin * rebin]
```

### If oscillation weighting is the issue:
- Nadir PDF not summing to 1.0
- combine_day_night applying weights twice
- Backend switch (FILE → NUFAST) producing different normalizations

### If signal templates are the issue:
- Templates never regenerated (timestamp check)
- Exposure/detector mass calculation error
- Energy binning mismatch between oscillation and templates

## Next Steps

1. **Run all tests:**
   ```bash
   bash run_diagnostics.sh 2>&1 | tee diagnostic_results.txt
   ```

2. **Identify which component fails:**
   - Look for "FAIL" or "100x" ratio

3. **If rebinning is the problem:**
   - Verify the lib/dataframe.py fix is present
   - Regenerate all signal templates
   - Re-run sensitivity analysis

4. **If oscillation is the problem:**
   - Check git changes to oscillation.py
   - Review nadir PDF handling
   - May need to revert to file backend

5. **Report findings:**
   - Which test showed 100x ratio?
   - What's the exact ratio value?
   - Which commit introduced the change?

## Files Generated

- `run_diagnostics.sh` - Master runner
- `diagnostic_signal_check.sh` - Signal template stats
- `test_rebin_logic.py` - Rebinning validation
- `test_oscillation_weighting.py` - Oscillation normalization
- `test_weighting_regression.py` - Git regression detection
- `compare_weighting_git.sh` - Show code changes
- `compare_signal_versions.py` - Old vs new comparison
- `DIAGNOSTICS_GUIDE.md` - This file
