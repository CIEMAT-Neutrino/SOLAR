#!/bin/bash
# Master diagnostic runner - checks all potential 100x sources

set -e
cd /pc/choozdsk01/users/manthey/SOLAR

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║               SIGNAL 100x ERROR DIAGNOSTIC SUITE                   ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Check rebin fix
echo "1️⃣  CHECKING REBIN_HIST2D FIX..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash diagnostic_signal_check.sh
echo ""

# 2. Test rebin logic
echo "2️⃣  TESTING REBIN LOGIC..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 test_rebin_logic.py
echo ""

# 3. Test oscillation weighting
echo "3️⃣  TESTING OSCILLATION WEIGHTING..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 test_oscillation_weighting.py
echo ""

# 4. Compare versions (optional)
echo "4️⃣  COMPARING WITH PREVIOUS VERSION..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "/path/to/old/SENSITIVITY" ]; then
    python3 compare_signal_versions.py
else
    echo "ℹ️  No old data path configured"
    echo "   To enable comparison, update compare_signal_versions.py with old path"
fi
echo ""

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                     DIAGNOSTICS COMPLETE                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "NEXT STEPS:"
echo "1. Review all outputs above for ✗ FAIL or ✗ ERROR"
echo "2. If rebin ratio is ~100x: rebinning bug still exists"
echo "3. If oscillation ratio is ~100x: weighting bug elsewhere"
echo "4. Check signal template timestamps to confirm regeneration"
echo ""
