# PDF Export Robustness Improvements

## Overview

The PDF export pipeline has been hardened to prevent hangs and timeouts that were occurring during presentation generation. This document describes the improvements made to ensure reliable, predictable PDF export behavior.

## Problem Statement

Previous PDF export attempts would hang indefinitely in certain scenarios:

- Browser process would hang during PDF printing
- Python subprocess had no timeout protection
- Error messages were unclear when failures occurred
- Browser selection didn't prioritize repo-local tools
- No validation that the selected browser was executable

## Solutions Implemented

### 1. Shell Script (`scripts/marp-pdf.sh`) Improvements

#### Browser Detection & Validation

- **Enhanced priority order**: Checks environment variables first, then repo-local paths (.chrome, .tools), then system browsers
- **Explicit validation**: Verifies selected browser is executable before use
- **Clear error messages**: Shows all candidates checked when no browser found
- **Added `--verbose` flag**: Enables debugging output to see browser selection process

#### Timeout Protection

- **Step-level timeouts**: 90s for Marp HTML rendering, 240s for Chrome PDF printing
- **360s overall script timeout**: Prevents indefinite hangs at any step
- **Timeout detection**: Distinguishes between timeout (124 exit code) and other failures

#### Additional Browser Flags

- Added flags for better container compatibility:
  - `--disable-background-networking`
  - `--disable-background-timer-throttling`
  - `--disable-client-side-phishing-detection`
  - `--disable-sync`
  - `--no-first-run`

#### File Validation

- Validates HTML file exists and has content before PDF printing
- Validates PDF file exists after printing
- Reports file sizes for debugging

### 2. Python Wrapper (`scripts/presentation_common.py`) Improvements

#### Overall Timeout Management

```python
def export_marp_pdf(markdown_path):
    """Export with overall 330s timeout wrapper"""
    overall_timeout = 330
    start_time = time.time()
    # ... call export_marp_pdf_internal() ...
    # Catches TimeoutExpired and wraps with helpful message
```

#### Better Error Handling

- Wraps subprocess TimeoutExpired with helpful guidance about browser setup
- Suggests increasing timeout or setting MARP_BROWSER_PATH
- Includes timing information in error messages
- Catches unexpected exceptions and includes execution time

#### Enhanced _run_export Function

- Increased timeout to 300s with clear parameter
- Added command logging for debugging
- Better error message prioritization
- Detailed timeout vs failure distinction

### 3. Environment Variables

The following environment variables control PDF export behavior:

```bash
# Override default browser executable
export MARP_BROWSER_PATH=/path/to/chrome

# Alternative browser path
export BROWSER_PATH=/path/to/chromium

# Set tools directory for bootstrap artifacts
export SOLAR_TOOLS_DIR=/path/to/tools
```

### 4. Test Suite

New robustness test suite: `tests/test_pdf_export_robustness.py`

Tests validate:

- PDF export enabled by default ✓
- Browser detection logic ✓
- Timeout handling and error messages ✓
- Shell script robustness ✓
- Error message clarity ✓

Run tests with:

```bash
python3 tests/test_pdf_export_robustness.py
```

## Timeout Architecture

The PDF export process now has **three layers** of timeout protection:

```
┌─ Python Wrapper (330s overall)
│  │
│  ├─ subprocess.run(..., timeout=300s)
│  │  │
│  │  └─ bash timeout command
│  │     │
│  │     ├─ timeout 90s (Marp HTML render)
│  │     └─ timeout 240s (Chrome PDF print)
```

This ensures:

1. **Python layer**: Catches any subprocess hang at 300s
2. **Bash layer**: Kills marp or chrome if they exceed step timeouts
3. **Overall wrapper**: Aborts if total export exceeds 330s

## Browser Selection Strategy

The browser selection now follows this priority:

1. `MARP_BROWSER_PATH` environment variable (if executable)
2. `BROWSER_PATH` environment variable (if executable)
3. Repo-local Chrome: `.chrome/chrome-linux64/chrome`
4. Repo-local Chrome-for-Testing: `.tools/chrome-for-testing-*/chrome-linux64/chrome`
5. System browsers: `google-chrome-stable`, `google-chrome`, `chromium`, `chromium-browser`

This ensures repo-local tools are preferred (faster, no version conflicts) while allowing system fallbacks.

## Debugging

### Verbose Mode

Run shell script with `--verbose` to see:

- Which browser was selected
- Browser executable path and source
- Step-by-step progress (HTML rendering, PDF printing)
- File sizes and timing

```bash
scripts/marp-pdf.sh output/presentations/MySlides.md --verbose
```

### Direct Script Testing

```bash
# Test browser detection without full export
bash -c '. scripts/marp-pdf.sh --verbose'

# Test with custom browser
MARP_BROWSER_PATH=/usr/bin/chrome scripts/marp-pdf.sh output/presentations/MySlides.md
```

### Python-Level Debugging

Edit `presentation_common.py` line ~164 to uncomment command logging:

```python
# print(f'[DEBUG] Running: {cmd_str}', file=sys.stderr)
```

## Edge Cases Handled

| Scenario | Handling |
|----------|----------|
| Browser doesn't exist | Error before subprocess, clear message |
| Browser not executable | Error before subprocess, clear message |
| Browser hangs (timeout) | Exit code 124, timeout message suggests fixes |
| Marp fails | Clear error from stdout/stderr |
| HTML generation times out | 90s timeout, clear error |
| PDF printing times out | 240s timeout, troubleshooting suggestions |
| Disk full | Subprocess error caught and reported |
| Memory exhaustion | Process killed by OS, error reported |
| No browser available | Informative error with candidates list |

## Performance Expectations

| Step | Expected Duration | Max Timeout |
|------|-------------------|------------|
| Marp HTML render | 30-60s | 90s |
| Chrome startup | 10-30s | - |
| PDF printing | 5-20s | 240s |
| **Total** | **45-110s** | **330s** |

If export takes > 300s, suspect:

- Insufficient memory (Chrome needs 500MB+)
- Disk I/O issues
- Browser binary corrupted
- Network timeout during bootstrap

## Troubleshooting

### "PDF printing timed out after 240 seconds"

**Cause**: Chrome process hung or system overloaded
**Fix**:

1. Check available memory: `free -h`
2. Try manual export: `scripts/marp-pdf.sh output/presentations/MySlides.md --verbose`
3. Set explicit browser: `export MARP_BROWSER_PATH=/usr/bin/chromium`
4. Increase timeout if needed (edit script line 80 and 147)

### "No Chromium-compatible browser found"

**Cause**: No browser installed or found
**Fix**:

1. Install Chromium: `sudo apt install chromium-browser`
2. Or set environment variable: `export MARP_BROWSER_PATH=/path/to/chrome`
3. Or let script bootstrap Chrome-for-Testing automatically

### "HTML rendering failed"

**Cause**: Marp CLI not found or invalid markdown
**Fix**:

1. Verify Marp installed: `which marp`
2. Check markdown syntax
3. Look at error message for specific issue

## Future Improvements

Potential enhancements for even better robustness:

1. **Process monitoring**: Track browser CPU/memory during export
2. **Retry logic**: Automatic retry with different browser on timeout
3. **Partial output recovery**: Save intermediate HTML if PDF fails
4. **Caching**: Cache generated HTML to avoid re-rendering on retry
5. **Health checks**: Verify browser works before starting export
6. **Graceful degradation**: Fall back to HTML-only output if PDF fails

