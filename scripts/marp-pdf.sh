#!/usr/bin/env bash
set -euo pipefail

# Force Marp export to PDF using HTML rendering plus headless Chromium.
# Usage:
#   scripts/marp-pdf.sh input.md [output.pdf]
# Example:
#   scripts/marp-pdf.sh presentations/TruncatedSensitivitySignificanceWorkflow.md

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 input.md [output.pdf]" >&2
  exit 1
fi

input="$1"
if [[ ! -f "$input" ]]; then
  echo "Input file not found: $input" >&2
  exit 1
fi

if [[ $# -eq 2 ]]; then
  output="$2"
else
  output="${input%.md}.pdf"
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_html="${input%.md}.html"

if ! command -v marp >/dev/null 2>&1; then
  echo "Marp CLI is not available in this environment." >&2
  exit 1
fi

browser=""
if [[ -n "${MARP_BROWSER_PATH:-}" && -x "${MARP_BROWSER_PATH}" ]]; then
  browser="$MARP_BROWSER_PATH"
elif compgen -G "$root_dir/.cache/ms-playwright/chromium-*/chrome-linux/chrome" >/dev/null; then
  browser="$(ls -1 "$root_dir"/.cache/ms-playwright/chromium-*/chrome-linux/chrome 2>/dev/null | sort | tail -n 1)"
elif command -v google-chrome-stable >/dev/null 2>&1; then
  browser="$(command -v google-chrome-stable)"
elif command -v google-chrome >/dev/null 2>&1; then
  browser="$(command -v google-chrome)"
elif command -v chromium >/dev/null 2>&1; then
  browser="$(command -v chromium)"
elif command -v chromium-browser >/dev/null 2>&1; then
  browser="$(command -v chromium-browser)"
fi

marp --html --allow-local-files -o "$tmp_html" "$input"

if [[ -z "$browser" ]]; then
  echo "No Chromium-compatible browser found for PDF printing." >&2
  echo "Set MARP_BROWSER_PATH or install Chromium/Chrome." >&2
  exit 1
fi

"$browser" \
  --headless \
  --no-sandbox \
  --disable-gpu \
  --disable-dev-shm-usage \
  --print-to-pdf="$output" \
  "file://$tmp_html"

echo "Exported PDF: $output"