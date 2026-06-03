#!/bin/bash
# Compare oscillation weighting logic between working and current versions

echo "════════════════════════════════════════════════════════════════════════════"
echo "GIT COMPARISON: OSCILLATION WEIGHTING LOGIC"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

WORKING_COMMIT="bb5d270"  # Last commit before d41cee2 refactor
CURRENT_COMMIT="HEAD"

cd /pc/choozdsk01/users/manthey/SOLAR

echo "Comparing between:"
echo "  Working:  $WORKING_COMMIT ($(git log -1 --format=%s $WORKING_COMMIT))"
echo "  Current:  $CURRENT_COMMIT"
echo ""

# Find old oscillation functions
echo "════════════════════════════════════════════════════════════════════════════"
echo "1. OLD process_oscillation_map() (from $WORKING_COMMIT)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

git show $WORKING_COMMIT:lib/lib_osc.py 2>/dev/null | grep -A 80 "def process_oscillation_map" | head -100

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "2. OLD get_nadir_angle() (from $WORKING_COMMIT)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

git show $WORKING_COMMIT:lib/lib_osc.py 2>/dev/null | grep -A 30 "def get_nadir_angle"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "3. CURRENT combine_day_night() and get_nadir_pdf functions"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

grep -A 20 "def combine_day_night" lib/oscillation_backends.py

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "4. KEY DIFFERENCES IN WEIGHTING"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

echo "OLD approach (process_oscillation_map):"
echo "  1. Load oscillation probability from ROOT file (2000-bin nadir)"
echo "  2. Load nadir PDF via get_nadir_angle() from nadir.root"
echo "  3. Interpolate nadir PDF to match oscillation nadir bins"
echo "  4. Normalize nadir PDF"
echo "  5. Apply: df.mul(nadir_y, axis=0) ← multiply rows by nadir weights"
echo ""

echo "NEW approach (combine_day_night):"
echo "  1. Compute oscillation probability (nufast/prob3)"
echo "  2. Load nadir PDF via get_nadir_pdf_file() or get_nadir_pdf_nufast()"
echo "  3. Already at analysis resolution (40 bins)"
echo "  4. Apply: weighted = night_df.values * nadir_pdf[:, np.newaxis]"
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "5. POTENTIAL ISSUES"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

echo "Issue 1: Interpolation difference"
echo "  OLD: Interpolates nadir PDF from 2000 bins to whatever oscillation resolution"
echo "  NEW: Computes or loads at analysis resolution (40 bins)"
echo "  Risk: Different interpolation methods → normalization change"
echo ""

echo "Issue 2: Backend switching"
echo "  OLD: Always used ROOT files (process_oscillation_map)"
echo "  NEW: Can use nufast/prob3 (different source of probabilities)"
echo "  Risk: New backends may produce different normalizations"
echo ""

echo "Issue 3: Energy grid changes"
echo "  Check: Did OSC_ENERGY_BINS or OSC_ENERGY_RANGE change?"
grep "OSC_ENERGY" analysis/physics.json 2>/dev/null || echo "  (Physics config not found)"
git show $WORKING_COMMIT:import/analysis.json 2>/dev/null | grep "OSC_ENERGY" || echo "  (Old config not accessible)"
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "6. FULL GIT DIFF FOR OSCILLATION FILES"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Differences in lib_osc.py (relevant section):"

git diff $WORKING_COMMIT HEAD -- lib/lib_osc.py lib/oscillation.py 2>/dev/null | \
    grep -A 5 -B 5 "nadir\|interp\|convolve\|process_oscillation" | head -150

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "To identify the 100x error source:"
echo ""
echo "1. Check if nadir PDF normalization changed:"
echo "   python3 test_oscillation_weighting.py"
echo ""
echo "2. Check if rebinning introduced the error:"
echo "   python3 test_rebin_logic.py"
echo ""
echo "3. Compare actual signal template values:"
echo "   python3 compare_signal_versions.py"
echo ""
echo "4. Identify which commit introduced the regression:"
echo "   git bisect start"
echo "   git bisect bad HEAD"
echo "   git bisect good bb5d270"
echo "   (Then run your sensitivity analysis on each midpoint)"
echo ""
