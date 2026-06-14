#!/usr/bin/env bash
set -euo pipefail

# Marp export to PDF via marp --pdf (Chrome CDP).
# Uses CDP — not Chrome --print-to-pdf — so page dimensions and orientation are correct.
# On first run with no Chrome found, downloads Chrome for Testing to .tools/chrome-linux64/.
# Usage: scripts/marp-pdf.sh input.md [output.pdf] [--verbose]

verbose=0
[[ $# -lt 1 ]] && { echo "Usage: $0 input.md [output.pdf] [--verbose]" >&2; exit 1; }

input="$1"; shift
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --verbose) verbose=1 ;;
    *) [[ -z "$output" ]] && output="$1" ;;
  esac
  shift
done

[[ -f "$input" ]] || { echo "ERROR: Input not found: $input" >&2; exit 1; }
[[ -z "$output" ]] && output="${input%.md}.pdf"

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tools_dir="$root_dir/.tools"

log_info() { [[ $verbose -eq 1 ]] && echo "[INFO] $*" >&2 || true; }
log_error() { echo "[ERROR] $*" >&2; }

# ── marp binary ───────────────────────────────────────────────────────────────
# Prefer standalone marp (bundles Node 18+); system marp runs on Node 16 which lacks ReadableStream
marp_bin="marp"
local_marp="$(ls -1 "$tools_dir"/marp-cli-v*-linux/marp 2>/dev/null | sort | tail -n 1 || true)"
[[ -x "$local_marp" ]] && marp_bin="$local_marp"

if ! command -v "$marp_bin" >/dev/null 2>&1 && [[ ! -x "$marp_bin" ]]; then
  log_error "Marp CLI not found."
  exit 1
fi

# ── Chrome discovery + one-time bootstrap ─────────────────────────────────────
bootstrap_chrome() {
  local chrome_dir="$tools_dir/chrome-linux64"
  local chrome_bin="$chrome_dir/chrome"
  [[ -x "$chrome_bin" ]] && echo "$chrome_bin" && return 0

  log_info "Downloading Chrome for Testing to $chrome_dir ..."
  mkdir -p "$tools_dir"

  local info
  info="$(curl -fsSL --max-time 30 \
    'https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json' \
    2>/dev/null)" || { log_error "Could not fetch Chrome download info (no network?)"; return 1; }

  local url
  url="$(python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print(next(x['url'] for x in d['channels']['Stable']['downloads']['chrome'] if x['platform'] == 'linux64'))
" <<< "$info")" || { log_error "Could not parse Chrome download URL"; return 1; }

  local archive="$tools_dir/chrome-linux64.zip"
  curl -fsSL "$url" -o "$archive" || { log_error "Failed to download Chrome archive"; return 1; }

  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$archive" -d "$tools_dir"
  else
    python3 -c "
import zipfile, sys
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(sys.argv[2])
" "$archive" "$tools_dir"
  fi
  rm -f "$archive"

  [[ -x "$chrome_bin" ]] || chmod +x "$chrome_bin" 2>/dev/null || true
  [[ -x "$chrome_bin" ]] || { log_error "Chrome binary not found at $chrome_bin after extraction"; return 1; }
  log_info "Chrome installed: $chrome_bin"
  echo "$chrome_bin"
}

find_chrome() {
  for env_var in MARP_BROWSER_PATH BROWSER_PATH; do
    local val="${!env_var:-}"
    # Skip wrapper scripts — we create our own wrapper below
    [[ -n "$val" && -x "$val" && "$val" != *chrome-wrapper* ]] && echo "$val" && return 0
  done
  for glob_match in \
      "$tools_dir/chrome-linux64/chrome" \
      "$tools_dir"/chrome-for-testing-*/chrome-linux64/chrome; do
    [[ -x "$glob_match" && "$glob_match" != *chrome-wrapper* ]] && echo "$glob_match" && return 0
  done
  for sys_chrome in google-chrome-stable google-chrome chromium chromium-browser; do
    local bin
    bin="$(command -v "$sys_chrome" 2>/dev/null || true)"
    [[ -n "$bin" && -x "$bin" ]] && echo "$bin" && return 0
  done
  return 1
}

real_browser="$(find_chrome 2>/dev/null || true)"
if [[ -z "$real_browser" ]]; then
  log_info "No Chrome found; attempting one-time download of Chrome for Testing..."
  real_browser="$(bootstrap_chrome)" || {
    log_error "Chrome not available and download failed. Install Chrome/Chromium or set MARP_BROWSER_PATH."
    exit 1
  }
fi

# Wrap the Chrome binary to inject container-required flags.
# marp launches Chrome via --browser-path; flags added here apply in all environments.
chrome_wrapper="$tools_dir/chrome-wrapper.sh"
cat > "$chrome_wrapper" << EOF
#!/usr/bin/env bash
exec "$real_browser" --no-sandbox --disable-dev-shm-usage --disable-gpu "\$@"
EOF
chmod +x "$chrome_wrapper"
browser="$chrome_wrapper"

log_info "marp: $marp_bin"
log_info "browser (real): $real_browser"
log_info "browser (wrapper): $browser"
export BROWSER_PATH="$real_browser"

# ── marp --pdf via Chrome CDP ─────────────────────────────────────────────────
# CDP (not --print-to-pdf): respects @page CSS dimensions, correct orientation, no rotation fix needed.
# --browser-timeout 60: prevents hang after Chrome closes its CDP connection.
marp_out=""
marp_config="$root_dir/src/tools/marp.config.js"
if ! marp_out="$("$marp_bin" --pdf --allow-local-files --browser-path "$browser" \
    --browser-timeout 60 --config "$marp_config" -o "$output" "$input" 2>&1)"; then
  # Show ERROR/FAIL lines first; fall back to full output
  error_lines="$(echo "$marp_out" | grep -iE '\[ *ERROR\]|\[ *FAIL' || true)"
  if [[ -z "$error_lines" ]]; then
    error_lines="$(echo "$marp_out" | grep -v '^\s*$' | grep -iv '\[ *INFO\]\|\[ *WARN\]' | head -3 || true)"
  fi
  [[ -z "$error_lines" ]] && error_lines="$(echo "$marp_out" | tail -3)"
  log_error "marp --pdf failed:"
  while IFS= read -r line; do
    [[ -n "$line" ]] && log_error "  $line"
  done <<< "$error_lines"
  exit 1
fi

if [[ ! -f "$output" || $(stat -c%s "$output" 2>/dev/null || echo 0) -le 0 ]]; then
  log_error "marp --pdf produced empty output"
  exit 1
fi

log_info "marp --pdf succeeded"
echo "Exported PDF: $output"
